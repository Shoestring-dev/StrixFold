//! A tokenizer's vocabulary as the bytes each token writes, for constrained draws: byte-level BPE entries decoded
//! through GPT-2's byte map, added tokens as their content, special tokens (BOS, EOS, pads, image spans) apart: no
//! grammar text, only EOS where a grammar may end. Tries over the tokens let a mask be found by walking the grammar
//! once per shared prefix; tokens that are plain text inside a JSON string (or a DSML raw value) are kept apart, so
//! inside one only the rest are walked.
const std = @import("std");
const machine = @import("machine.zig");
const Allocator = std.mem.Allocator;

pub const Node = struct { edge0: u32, nedge: u32, tok0: u32, ntok: u32 };
pub const Edge = struct { byte: u8, next: u32 };

/// Tokens by their bytes: node 0 is the root (the empty text).
pub const Trie = struct {
    nodes: []const Node = &.{},
    edges: []const Edge = &.{},
    toks: []const u32 = &.{},
};

pub const Error = error{ Unsupported, OutOfMemory };

/// Plain JSON string tokens by their characters, below this many: one bitset each (a string near its maxLength).
pub const short_chars = 17;

pub const Vocab = struct {
    arena: std.heap.ArenaAllocator,
    /// Token ids 0 .. size - 1.
    size: u32,
    offs: []const u32, // token t's bytes: bytes[offs[t]..offs[t + 1]]
    bytes: []const u8,
    special: []const bool, // a special token: never grammar text
    all: Trie,
    /// The tokens that are not plain text in a JSON string, and in a DSML raw value.
    json_rest: Trie,
    raw_rest: Trie,
    /// Plain text in a JSON string (complete UTF-8, no control byte, quote or backslash): all of it, those of at most
    /// r characters (r < short_chars), and the longer ones with their characters.
    json_plain_bits: []const u64,
    json_plain_upto: []const []const u64,
    json_long: []const u32,
    json_long_chars: []const u32,
    json_plain_most: u32, // the most characters a json_plain token holds
    /// Plain text in a DSML raw value (complete UTF-8 without '<' or '/').
    raw_plain_bits: []const u64,
    longest: u32,

    pub fn text(v: *const Vocab, t: u32) []const u8 {
        return v.bytes[v.offs[t]..v.offs[t + 1]];
    }

    pub fn words(v: *const Vocab) usize {
        return (v.size + 63) / 64;
    }

    pub fn deinit(v: *Vocab, gpa: Allocator) void {
        var arena = v.arena;
        arena.deinit();
        gpa.destroy(v);
    }
};

/// GPT-2's bytes_to_unicode: the code point each byte is written as in a byte-level BPE vocabulary.
fn byteChars() [256]u21 {
    var out: [256]u21 = undefined;
    var n: u21 = 0;
    for (0..256) |b| {
        const printable = (b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE and b <= 0xFF);
        if (printable) {
            out[b] = @intCast(b);
        } else {
            out[b] = 256 + n;
            n += 1;
        }
    }
    return out;
}

/// The bytes a byte-level BPE entry stands for, or null when it uses a code point outside the byte map.
fn decodeEntry(a: Allocator, inverse: *const std.AutoHashMapUnmanaged(u21, u8), entry: []const u8) Error!?[]const u8 {
    var out: std.ArrayList(u8) = .empty;
    const view = std.unicode.Utf8View.init(entry) catch return null;
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| try out.append(a, inverse.get(cp) orelse return null);
    return out.items;
}

/// The vocabulary of a tokenizer.json (a byte-level BPE model); error.Unsupported for any other kind.
pub fn parse(gpa: Allocator, json_text: []const u8) Error!*Vocab {
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    const s = scratch.allocator();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, s, json_text, .{}) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.Unsupported,
    };
    if (doc != .object) return error.Unsupported;
    const model = doc.object.get("model") orelse return error.Unsupported;
    if (model != .object) return error.Unsupported;
    const kind = model.object.get("type") orelse return error.Unsupported;
    if (kind != .string or !std.mem.eql(u8, kind.string, "BPE")) return error.Unsupported;
    const decoder = doc.object.get("decoder") orelse return error.Unsupported;
    if (decoder != .object) return error.Unsupported;
    const dkind = decoder.object.get("type") orelse return error.Unsupported;
    if (dkind != .string or !std.mem.eql(u8, dkind.string, "ByteLevel")) return error.Unsupported;
    const vocab = model.object.get("vocab") orelse return error.Unsupported;
    if (vocab != .object) return error.Unsupported;

    const chars = byteChars();
    var inverse: std.AutoHashMapUnmanaged(u21, u8) = .empty;
    for (chars, 0..) |c, b| try inverse.put(s, c, @intCast(b));

    // id -> bytes (null: none yet) and special flags
    var size: u32 = 0;
    for (vocab.object.values()) |id| {
        if (id != .integer or id.integer < 0 or id.integer >= 1 << 24) return error.Unsupported;
        size = @max(size, @as(u32, @intCast(id.integer)) + 1);
    }
    const added = doc.object.get("added_tokens");
    if (added) |ad| if (ad == .array) for (ad.array.items) |t| {
        const id = if (t == .object) t.object.get("id") else null;
        if (id == null or id.? != .integer or id.?.integer < 0 or id.?.integer >= 1 << 24) return error.Unsupported;
        size = @max(size, @as(u32, @intCast(id.?.integer)) + 1);
    };
    const texts = try s.alloc(?[]const u8, size);
    @memset(texts, null);
    const special_s = try s.alloc(bool, size);
    @memset(special_s, false);
    for (vocab.object.keys(), vocab.object.values()) |k, id| texts[@intCast(id.integer)] = try decodeEntry(s, &inverse, k);
    if (added) |ad| if (ad == .array) for (ad.array.items) |t| {
        const id: u32 = @intCast(t.object.get("id").?.integer);
        const content = t.object.get("content") orelse return error.Unsupported;
        if (content != .string) return error.Unsupported;
        const special = if (t.object.get("special")) |sp| sp == .bool and sp.bool else false;
        special_s[id] = special;
        texts[id] = if (special) null else content.string;
    };

    const v = try gpa.create(Vocab);
    errdefer gpa.destroy(v);
    v.arena = .init(gpa);
    errdefer v.arena.deinit();
    const a = v.arena.allocator();
    v.size = size;
    var total: usize = 0;
    for (texts) |t| total += if (t) |x| x.len else 0;
    const offs = try a.alloc(u32, size + 1);
    const bytes = try a.alloc(u8, total);
    const special = try a.alloc(bool, size);
    var at: u32 = 0;
    var longest: u32 = 0;
    for (texts, 0..) |t, i| {
        offs[i] = at;
        // an id with no text (a gap, a special token, an entry outside the byte map) is never grammar text
        special[i] = special_s[i] or t == null or t.?.len == 0;
        if (t) |x| if (!special[i]) {
            @memcpy(bytes[at..][0..x.len], x);
            at += @intCast(x.len);
            longest = @max(longest, @as(u32, @intCast(x.len)));
        };
    }
    offs[size] = at;
    v.offs = offs;
    v.bytes = bytes[0..at];
    v.special = special;
    v.longest = longest;
    try classify(a, v);
    return v;
}

/// Reads `dir`/tokenizer.json.
pub fn load(gpa: Allocator, io: std.Io, dir: []const u8) !*Vocab {
    const path = try std.fs.path.join(gpa, &.{ dir, "tokenizer.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(256 << 20));
    defer gpa.free(text);
    return parse(gpa, text);
}

/// Each token's plain-text kinds and the tries.
fn classify(a: Allocator, v: *Vocab) Error!void {
    const words = v.words();
    const json_bits = try a.alloc(u64, words);
    const raw_bits = try a.alloc(u64, words);
    @memset(json_bits, 0);
    @memset(raw_bits, 0);
    const upto = try a.alloc([]u64, short_chars);
    for (upto) |*u| {
        u.* = try a.alloc(u64, words);
        @memset(u.*, 0);
    }
    var json_long: std.ArrayList(u32) = .empty;
    var json_long_chars: std.ArrayList(u32) = .empty;
    var json_rest: std.ArrayList(u32) = .empty;
    var raw_rest: std.ArrayList(u32) = .empty;
    var everything: std.ArrayList(u32) = .empty;
    var most: u32 = 0;
    for (0..v.size) |i| {
        const t: u32 = @intCast(i);
        if (v.special[t]) continue;
        try everything.append(a, t);
        const x = v.text(t);
        var k: u8 = 0;
        var ok = true;
        var json_ok = true;
        var raw_ok = true;
        var n: u32 = 0;
        for (x) |b| {
            if (k == 0) n += 1;
            if (!machine.utf8(&k, b)) {
                ok = false;
                break;
            }
            if (b < 0x20 or b == '"' or b == '\\') json_ok = false;
            if (b == '<' or b == '/') raw_ok = false;
        }
        ok = ok and k == 0;
        if (ok and json_ok) {
            const bit = @as(u64, 1) << @intCast(t % 64);
            json_bits[t / 64] |= bit;
            most = @max(most, n);
            if (n < short_chars) {
                for (upto[n..]) |u| u[t / 64] |= bit;
            } else {
                try json_long.append(a, t);
                try json_long_chars.append(a, n);
            }
        } else try json_rest.append(a, t);
        if (ok and raw_ok) raw_bits[t / 64] |= @as(u64, 1) << @intCast(t % 64) else try raw_rest.append(a, t);
    }
    v.json_plain_bits = json_bits;
    v.json_plain_upto = upto;
    v.json_long = json_long.items;
    v.json_long_chars = json_long_chars.items;
    v.json_plain_most = most;
    v.raw_plain_bits = raw_bits;
    v.all = try build(a, v, everything.items);
    v.json_rest = try build(a, v, json_rest.items);
    v.raw_rest = try build(a, v, raw_rest.items);
}

const Builder = struct {
    a: Allocator,
    v: *const Vocab,
    nodes: std.ArrayList(Node) = .empty,
    edges: std.ArrayList(Edge) = .empty,
    toks: std.ArrayList(u32) = .empty,

    /// The node for sorted[lo..hi], every token there sharing its first `d` bytes; returns its index.
    fn node(b: *Builder, sorted: []const u32, d: usize) Error!u32 {
        const id: u32 = @intCast(b.nodes.items.len);
        try b.nodes.append(b.a, undefined);
        var i: usize = 0;
        const tok0: u32 = @intCast(b.toks.items.len);
        while (i < sorted.len and b.v.text(sorted[i]).len == d) : (i += 1) try b.toks.append(b.a, sorted[i]);
        const ntok: u32 = @intCast(b.toks.items.len - tok0);
        // the groups by the byte at d, their edges together
        var groups: usize = 0;
        var j = i;
        while (j < sorted.len) {
            const c = b.v.text(sorted[j])[d];
            while (j < sorted.len and b.v.text(sorted[j])[d] == c) j += 1;
            groups += 1;
        }
        const edge0: u32 = @intCast(b.edges.items.len);
        try b.edges.appendNTimes(b.a, undefined, groups);
        var g: usize = 0;
        j = i;
        while (j < sorted.len) : (g += 1) {
            const c = b.v.text(sorted[j])[d];
            const start = j;
            while (j < sorted.len and b.v.text(sorted[j])[d] == c) j += 1;
            const child = try b.node(sorted[start..j], d + 1);
            b.edges.items[edge0 + g] = .{ .byte = c, .next = child };
        }
        b.nodes.items[id] = .{ .edge0 = edge0, .nedge = @intCast(groups), .tok0 = tok0, .ntok = ntok };
        return id;
    }
};

fn build(a: Allocator, v: *const Vocab, tokens: []u32) Error!Trie {
    std.mem.sort(u32, tokens, v, struct {
        fn less(vv: *const Vocab, x: u32, y: u32) bool {
            return switch (std.mem.order(u8, vv.text(x), vv.text(y))) {
                .lt => true,
                .gt => false,
                .eq => x < y,
            };
        }
    }.less);
    var b: Builder = .{ .a = a, .v = v };
    _ = try b.node(tokens, 0);
    return .{ .nodes = b.nodes.items, .edges = b.edges.items, .toks = b.toks.items };
}

/// A small byte-level vocabulary for tests: the 256 bytes as ids 0-255, then `extra` texts, then `specials` special
/// tokens (each with no text).
pub fn forTest(gpa: Allocator, extra: []const []const u8, specials: u32) !*Vocab {
    const v = try gpa.create(Vocab);
    errdefer gpa.destroy(v);
    v.arena = .init(gpa);
    errdefer v.arena.deinit();
    const a = v.arena.allocator();
    const size: u32 = @intCast(256 + extra.len + specials);
    var bytes: std.ArrayList(u8) = .empty;
    const offs = try a.alloc(u32, size + 1);
    const special = try a.alloc(bool, size);
    var longest: u32 = 1;
    for (0..size) |i| {
        offs[i] = @intCast(bytes.items.len);
        special[i] = i >= 256 + extra.len;
        if (i < 256) try bytes.append(a, @intCast(i)) else if (i < 256 + extra.len) {
            try bytes.appendSlice(a, extra[i - 256]);
            longest = @max(longest, @as(u32, @intCast(extra[i - 256].len)));
        }
    }
    offs[size] = @intCast(bytes.items.len);
    v.size = size;
    v.offs = offs;
    v.bytes = bytes.items;
    v.special = special;
    v.longest = longest;
    try classify(a, v);
    return v;
}

test "a byte-level tokenizer.json decodes through the byte map; special tokens stay apart" {
    const gpa = std.testing.allocator;
    // "Ġ" is the space byte, "Ċ" the newline; id 3 has a code point outside the map
    const text =
        \\{"model": {"type": "BPE", "vocab": {"a": 0, "Ġb": 1, "Ċ": 2, "一": 3}, "merges": []},
        \\ "decoder": {"type": "ByteLevel"},
        \\ "added_tokens": [{"id": 4, "content": "<eos>", "special": true}, {"id": 5, "content": "</think>", "special": false}]}
    ;
    const v = try parse(gpa, text);
    defer v.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 6), v.size);
    try std.testing.expectEqualStrings(" b", v.text(1));
    try std.testing.expectEqualStrings("\n", v.text(2));
    try std.testing.expect(v.special[3] and v.special[4] and !v.special[5]);
    try std.testing.expectEqualStrings("</think>", v.text(5));
    // "\n" is not plain JSON string text; " b" is
    try std.testing.expect(v.json_plain_bits[0] & (1 << 1) != 0);
    try std.testing.expect(v.json_plain_bits[0] & (1 << 2) == 0);
    // the trie holds every non-special token once
    try std.testing.expectEqual(@as(usize, 4), v.all.toks.len);
}
