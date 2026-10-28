//! guided_regex and JSON schema `pattern` as a byte DFA: a Python-style subset (literals, classes of ASCII ranges, `.`,
//! \d \w \s and their negations, groups, |, * + ? {m,n}) over UTF-8 text, built as a Thompson NFA and determinized.
//! Every state the DFA keeps still reaches a match, so a byte it takes never strands the reply. Negated classes and
//! `.` take every multi-byte character; anything else outside the subset (backreferences, lookaround, \b, non-ASCII
//! class ranges) is refused with a message.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const dead: u32 = std.math.maxInt(u32);

/// A refusal's words, kept without allocation.
pub const Diag = struct {
    buf: [240]u8 = undefined,
    len: usize = 0,

    pub fn set(d: *Diag, comptime fmt: []const u8, args: anytype) void {
        const out: []const u8 = std.fmt.bufPrint(&d.buf, fmt, args) catch &d.buf;
        d.len = out.len;
    }

    pub fn text(d: *const Diag) []const u8 {
        return d.buf[0..d.len];
    }
};

pub const Error = error{ Unsupported, OutOfMemory };

pub const Dfa = struct {
    classes: [256]u8,
    ncls: u32,
    next: []const u32, // state * ncls + class: the next state, or dead
    accept: []const bool,
    start: u32,

    pub fn step(d: *const Dfa, s: u32, b: u8) u32 {
        return d.next[s * d.ncls + d.classes[b]];
    }
};

/// full: the whole text matches (guided_regex); search: JSON schema's `pattern`, a match anywhere unless ^ or $ anchor it.
pub const Mode = enum { full, search };

const max_nfa = 40_000;
const max_dfa = 4096;
const max_count = 1000;

const Set = struct {
    ascii: std.StaticBitSet(128) = .empty,
    multi: bool = false, // every multi-byte UTF-8 character

    fn range(s: *Set, lo: u8, hi: u8) void {
        var c: usize = lo;
        while (c <= hi) : (c += 1) s.ascii.set(c);
    }

    fn negate(s: Set) Set {
        return .{ .ascii = s.ascii.complement(), .multi = !s.multi };
    }

    fn merge(s: *Set, o: Set) void {
        s.ascii.setUnion(o.ascii);
        s.multi = s.multi or o.multi;
    }
};

const Ast = union(enum) {
    set: Set,
    bytes: []const u8,
    cat: []const *const Ast,
    alt: []const *const Ast,
    rep: struct { sub: *const Ast, min: u32, max: ?u32 },
};

const Parser = struct {
    a: Allocator,
    src: []const u8,
    at: usize = 0,
    diag: *Diag,

    fn fail(p: *Parser, comptime fmt: []const u8, args: anytype) Error {
        p.diag.set(fmt, args);
        return error.Unsupported;
    }

    fn peek(p: *const Parser) ?u8 {
        return if (p.at < p.src.len) p.src[p.at] else null;
    }

    fn node(p: *Parser, v: Ast) Error!*const Ast {
        const n = try p.a.create(Ast);
        n.* = v;
        return n;
    }

    fn alt(p: *Parser) Error!*const Ast {
        var branches: std.ArrayList(*const Ast) = .empty;
        try branches.append(p.a, try p.cat());
        while (p.peek() == '|') {
            p.at += 1;
            try branches.append(p.a, try p.cat());
        }
        if (branches.items.len == 1) return branches.items[0];
        return p.node(.{ .alt = branches.items });
    }

    fn cat(p: *Parser) Error!*const Ast {
        var items: std.ArrayList(*const Ast) = .empty;
        while (p.peek()) |c| {
            if (c == '|' or c == ')') break;
            try items.append(p.a, try p.repeat());
        }
        if (items.items.len == 1) return items.items[0];
        return p.node(.{ .cat = items.items });
    }

    fn number(p: *Parser) ?u32 {
        const start = p.at;
        while (p.peek()) |c| {
            if (c < '0' or c > '9') break;
            p.at += 1;
        }
        if (p.at == start) return null;
        return std.fmt.parseInt(u32, p.src[start..p.at], 10) catch max_count + 1;
    }

    /// {m}, {m,}, {m,n} at `at` (null and nothing read when it is not one: a literal brace).
    fn braces(p: *Parser) Error!?[2]?u32 {
        const save = p.at;
        p.at += 1;
        const lo = p.number();
        var hi: ?u32 = lo;
        if (p.peek() == ',') {
            p.at += 1;
            hi = p.number();
        }
        if (lo == null or p.peek() != '}') {
            p.at = save;
            return null;
        }
        p.at += 1;
        return .{ lo, hi };
    }

    fn repeat(p: *Parser) Error!*const Ast {
        var sub = try p.atom();
        while (p.peek()) |c| {
            var min: u32 = 0;
            var max: ?u32 = null;
            switch (c) {
                '*' => p.at += 1,
                '+' => {
                    p.at += 1;
                    min = 1;
                },
                '?' => {
                    p.at += 1;
                    max = 1;
                },
                '{' => {
                    const b = try p.braces() orelse break;
                    min = b[0].?;
                    max = b[1];
                },
                else => break,
            }
            if (min > max_count or (max != null and (max.? > max_count or max.? < min))) return p.fail("regex repeat counts must be 0 to {d}, min not above max", .{max_count});
            if (p.peek() == '?') p.at += 1 // lazy: the same language
            else if (p.peek() == '+') return p.fail("possessive repeats are not supported in regex", .{});
            sub = try p.node(.{ .rep = .{ .sub = sub, .min = min, .max = max } });
        }
        return sub;
    }

    fn single(p: *Parser, c: u8) Error!*const Ast {
        var s: Set = .{};
        s.ascii.set(c);
        return p.node(.{ .set = s });
    }

    /// A non-ASCII literal character at `at` (its UTF-8 bytes).
    fn utf8(p: *Parser) Error!*const Ast {
        const n = std.unicode.utf8ByteSequenceLength(p.src[p.at]) catch return p.fail("regex is not valid UTF-8", .{});
        if (p.at + n > p.src.len) return p.fail("regex is not valid UTF-8", .{});
        const bytes = p.src[p.at..][0..n];
        _ = std.unicode.utf8Decode(bytes) catch return p.fail("regex is not valid UTF-8", .{});
        p.at += n;
        return p.node(.{ .bytes = bytes });
    }

    fn atom(p: *Parser) Error!*const Ast {
        const c = p.peek().?;
        switch (c) {
            '(' => {
                p.at += 1;
                if (p.peek() == '?') {
                    if (p.at + 1 < p.src.len and p.src[p.at + 1] == ':') {
                        p.at += 2;
                    } else if (p.at + 2 < p.src.len and p.src[p.at + 1] == 'P' and p.src[p.at + 2] == '<') {
                        const close = std.mem.indexOfScalarPos(u8, p.src, p.at, '>') orelse return p.fail("regex group name is not closed", .{});
                        p.at = close + 1;
                    } else return p.fail("regex lookaround and inline flags are not supported", .{});
                }
                const sub = try p.alt();
                if (p.peek() != ')') return p.fail("regex has an unclosed group", .{});
                p.at += 1;
                return sub;
            },
            '[' => return p.class(),
            '.' => {
                p.at += 1;
                var s: Set = .{ .multi = true };
                s.range(0, 127);
                s.ascii.unset('\n');
                return p.node(.{ .set = s });
            },
            '\\' => {
                p.at += 1;
                const e = p.peek() orelse return p.fail("regex ends with a backslash", .{});
                if (try p.shorthand(e)) |s| {
                    p.at += 1;
                    return p.node(.{ .set = s });
                }
                return p.escaped();
            },
            '^', '$' => return p.fail("regex anchors are supported only at its start and end", .{}),
            '*', '+', '?' => return p.fail("regex repeat at position {d} has nothing to repeat", .{p.at}),
            ')' => return p.fail("regex has an unmatched )", .{}),
            else => {
                if (c >= 0x80) return p.utf8();
                p.at += 1;
                return p.single(c);
            },
        }
    }

    /// \d \w \s and their negations as sets; null for any other escape.
    fn shorthand(p: *Parser, e: u8) Error!?Set {
        _ = p;
        var s: Set = .{};
        switch (std.ascii.toLower(e)) {
            'd' => s.range('0', '9'),
            'w' => {
                s.range('0', '9');
                s.range('a', 'z');
                s.range('A', 'Z');
                s.ascii.set('_');
            },
            's' => for (" \t\n\r\x0b\x0c") |w| s.ascii.set(w),
            else => return null,
        }
        return if (std.ascii.isUpper(e)) s.negate() else s;
    }

    /// The literal an escape at `at` names (after the backslash): one ASCII byte, or a \u character's bytes.
    fn escaped(p: *Parser) Error!*const Ast {
        const e = p.peek().?;
        p.at += 1;
        const simple: ?u8 = switch (e) {
            'n' => '\n',
            't' => '\t',
            'r' => '\r',
            'f' => 0x0c,
            'v' => 0x0b,
            '0' => 0,
            else => null,
        };
        if (simple) |b| return p.single(b);
        if (e == 'x' or e == 'u') {
            const n: usize = if (e == 'x') 2 else 4;
            if (p.at + n > p.src.len) return p.fail("regex \\{c} needs {d} hex digits", .{ e, n });
            const cp = std.fmt.parseInt(u21, p.src[p.at..][0..n], 16) catch return p.fail("regex \\{c} needs {d} hex digits", .{ e, n });
            p.at += n;
            if (cp < 0x80) return p.single(@intCast(cp));
            var buf: [4]u8 = undefined;
            const len = std.unicode.utf8Encode(cp, &buf) catch return p.fail("regex \\u{x} is not a character", .{cp});
            return p.node(.{ .bytes = try p.a.dupe(u8, buf[0..len]) });
        }
        if (std.ascii.isAlphanumeric(e)) return p.fail("regex escape \\{c} is not supported", .{e});
        if (e >= 0x80) return p.fail("regex escape of a non-ASCII character is not supported", .{});
        return p.single(e);
    }

    /// One class member's byte (after any backslash), or a shorthand set.
    const Member = union(enum) { byte: u8, set: Set };

    fn member(p: *Parser) Error!Member {
        const c = p.peek() orelse return p.fail("regex has an unclosed [", .{});
        if (c >= 0x80) return p.fail("regex classes take ASCII characters only", .{});
        p.at += 1;
        if (c != '\\') return .{ .byte = c };
        const e = p.peek() orelse return p.fail("regex has an unclosed [", .{});
        if (try p.shorthand(e)) |s| {
            p.at += 1;
            return .{ .set = s };
        }
        const lit = try p.escaped();
        if (lit.* != .set) return p.fail("regex classes take ASCII characters only", .{});
        return .{ .byte = @intCast(lit.set.ascii.findFirstSet().?) };
    }

    fn class(p: *Parser) Error!*const Ast {
        p.at += 1;
        var negated = false;
        if (p.peek() == '^') {
            negated = true;
            p.at += 1;
        }
        var s: Set = .{};
        var first = true;
        while (true) {
            const c = p.peek() orelse return p.fail("regex has an unclosed [", .{});
            if (c == ']' and !first) {
                p.at += 1;
                break;
            }
            first = false;
            const m = try p.member();
            switch (m) {
                .set => |x| s.merge(x),
                .byte => |lo| {
                    if (p.peek() == '-' and p.at + 1 < p.src.len and p.src[p.at + 1] != ']') {
                        p.at += 1;
                        const hi = switch (try p.member()) {
                            .byte => |h| h,
                            .set => return p.fail("regex class range ends in a shorthand", .{}),
                        };
                        if (hi < lo) return p.fail("regex class range {c}-{c} is reversed", .{ lo, hi });
                        s.range(lo, hi);
                    } else s.ascii.set(lo);
                },
            }
        }
        return p.node(.{ .set = if (negated) s.negate() else s });
    }
};

const Range = struct { lo: u8, hi: u8, to: u32 };

const Nfa = struct {
    a: Allocator,
    eps: std.ArrayList(std.ArrayList(u32)) = .empty,
    edges: std.ArrayList(std.ArrayList(Range)) = .empty,
    diag: *Diag,

    const Frag = struct { start: u32, end: u32 };

    fn state(n: *Nfa) Error!u32 {
        if (n.eps.items.len >= max_nfa) {
            n.diag.set("regex is too large (over {d} automaton states)", .{max_nfa});
            return error.Unsupported;
        }
        try n.eps.append(n.a, .empty);
        try n.edges.append(n.a, .empty);
        return @intCast(n.eps.items.len - 1);
    }

    fn link(n: *Nfa, from: u32, to: u32) Error!void {
        try n.eps.items[from].append(n.a, to);
    }

    fn edge(n: *Nfa, from: u32, lo: u8, hi: u8, to: u32) Error!void {
        try n.edges.items[from].append(n.a, .{ .lo = lo, .hi = hi, .to = to });
    }

    /// A path from `from` to `to` through one byte of each range.
    fn path(n: *Nfa, from: u32, to: u32, ranges: []const [2]u8) Error!void {
        var at = from;
        for (ranges, 0..) |r, i| {
            const next = if (i + 1 == ranges.len) to else try n.state();
            try n.edge(at, r[0], r[1], next);
            at = next;
        }
    }

    /// Every multi-byte UTF-8 character from `from` to `to` (RFC 3629's well-formed sequences).
    fn multibyte(n: *Nfa, from: u32, to: u32) Error!void {
        const c: [2]u8 = .{ 0x80, 0xBF };
        try n.path(from, to, &.{ .{ 0xC2, 0xDF }, c });
        try n.path(from, to, &.{ .{ 0xE0, 0xE0 }, .{ 0xA0, 0xBF }, c });
        try n.path(from, to, &.{ .{ 0xE1, 0xEC }, c, c });
        try n.path(from, to, &.{ .{ 0xED, 0xED }, .{ 0x80, 0x9F }, c });
        try n.path(from, to, &.{ .{ 0xEE, 0xEF }, c, c });
        try n.path(from, to, &.{ .{ 0xF0, 0xF0 }, .{ 0x90, 0xBF }, c, c });
        try n.path(from, to, &.{ .{ 0xF1, 0xF3 }, c, c, c });
        try n.path(from, to, &.{ .{ 0xF4, 0xF4 }, .{ 0x80, 0x8F }, c, c });
    }

    fn build(n: *Nfa, ast: *const Ast) Error!Frag {
        switch (ast.*) {
            .set => |s| {
                const f: Frag = .{ .start = try n.state(), .end = try n.state() };
                var c: usize = 0;
                while (c < 128) {
                    if (!s.ascii.isSet(c)) {
                        c += 1;
                        continue;
                    }
                    var e = c;
                    while (e + 1 < 128 and s.ascii.isSet(e + 1)) e += 1;
                    try n.edge(f.start, @intCast(c), @intCast(e), f.end);
                    c = e + 1;
                }
                if (s.multi) try n.multibyte(f.start, f.end);
                return f;
            },
            .bytes => |bs| {
                const start = try n.state();
                var at = start;
                for (bs) |b| {
                    const next = try n.state();
                    try n.edge(at, b, b, next);
                    at = next;
                }
                return .{ .start = start, .end = at };
            },
            .cat => |items| {
                var f = try n.build(items[0]);
                for (items[1..]) |it| {
                    const g = try n.build(it);
                    try n.link(f.end, g.start);
                    f.end = g.end;
                }
                return f;
            },
            .alt => |items| {
                const f: Frag = .{ .start = try n.state(), .end = try n.state() };
                for (items) |it| {
                    const g = try n.build(it);
                    try n.link(f.start, g.start);
                    try n.link(g.end, f.end);
                }
                return f;
            },
            .rep => |r| {
                const start = try n.state();
                var end = start;
                for (0..r.min) |_| {
                    const g = try n.build(r.sub);
                    try n.link(end, g.start);
                    end = g.end;
                }
                if (r.max) |max| {
                    const out = try n.state();
                    for (r.min..max) |_| {
                        try n.link(end, out);
                        const g = try n.build(r.sub);
                        try n.link(end, g.start);
                        end = g.end;
                    }
                    try n.link(end, out);
                    return .{ .start = start, .end = out };
                }
                const loop = try n.state();
                const g = try n.build(r.sub);
                try n.link(end, loop);
                try n.link(loop, g.start);
                try n.link(g.end, loop);
                return .{ .start = start, .end = loop };
            },
        }
    }

    /// The states `set` reaches through empty edges (sorted, without repeats), into `out`.
    fn closure(n: *const Nfa, a: Allocator, set: []const u32, out: *std.ArrayList(u32), seen: []bool) Error!void {
        out.clearRetainingCapacity();
        var stack: std.ArrayList(u32) = .empty;
        defer stack.deinit(a);
        for (set) |s| if (!seen[s]) {
            seen[s] = true;
            try stack.append(a, s);
        };
        while (stack.pop()) |s| {
            try out.append(a, s);
            for (n.eps.items[s].items) |t| if (!seen[t]) {
                seen[t] = true;
                try stack.append(a, t);
            };
        }
        for (out.items) |s| seen[s] = false;
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    }
};

/// `src` compiled for `mode` into `a` (the caller's arena: nothing is freed on its own).
pub fn compile(a: Allocator, src: []const u8, mode: Mode, diag: *Diag) Error!Dfa {
    var body = src;
    var anchored_start = mode == .full;
    var anchored_end = mode == .full;
    if (body.len > 0 and body[0] == '^') {
        body = body[1..];
        anchored_start = true;
    }
    if (body.len > 0 and body[body.len - 1] == '$') {
        var slashes: usize = 0;
        while (slashes + 1 < body.len and body[body.len - 2 - slashes] == '\\') slashes += 1;
        if (slashes % 2 == 0) {
            body = body[0 .. body.len - 1];
            anchored_end = true;
        }
    }
    var p: Parser = .{ .a = a, .src = body, .diag = diag };
    var ast: *const Ast = if (body.len == 0) try p.node(.{ .cat = &.{} }) else try p.alt();
    if (p.at != p.src.len) return p.fail("regex has an unmatched )", .{});
    // a search matches anywhere: any characters on the open sides
    var any: Set = .{ .multi = true };
    any.range(0, 127);
    const anything = try p.node(.{ .rep = .{ .sub = try p.node(.{ .set = any }), .min = 0, .max = null } });
    if (!anchored_start) ast = try p.node(.{ .cat = try a.dupe(*const Ast, &.{ anything, ast }) });
    if (!anchored_end) ast = try p.node(.{ .cat = try a.dupe(*const Ast, &.{ ast, anything }) });

    var n: Nfa = .{ .a = a, .diag = diag };
    const frag = if (ast.* == .cat and ast.cat.len == 0) blk: {
        const s = try n.state();
        break :blk Nfa.Frag{ .start = s, .end = s };
    } else try n.build(ast);
    return determinize(a, &n, frag, diag);
}

fn determinize(a: Allocator, n: *const Nfa, frag: Nfa.Frag, diag: *Diag) Error!Dfa {
    // byte classes: bytes no edge tells apart share a class
    var cut: [257]bool = @splat(false);
    cut[0] = true;
    for (n.edges.items) |es| for (es.items) |e| {
        cut[e.lo] = true;
        cut[@as(usize, e.hi) + 1] = true;
    };
    var classes: [256]u8 = undefined;
    var reps: [256]u8 = undefined;
    var ncls: u32 = 0;
    for (0..256) |b| {
        if (cut[b]) {
            reps[ncls] = @intCast(b);
            ncls += 1;
        }
        classes[b] = @intCast(ncls - 1);
    }
    const seen = try a.alloc(bool, n.eps.items.len);
    @memset(seen, false);
    var sets: std.ArrayList([]const u32) = .empty;
    var index: std.StringHashMapUnmanaged(u32) = .empty;
    var next: std.ArrayList(u32) = .empty;
    var accept: std.ArrayList(bool) = .empty;
    var scratch: std.ArrayList(u32) = .empty;
    var moved: std.ArrayList(u32) = .empty;

    try n.closure(a, &.{frag.start}, &scratch, seen);
    _ = try intern(a, &sets, &index, &accept, scratch.items, frag.end);
    var at: usize = 0;
    while (at < sets.items.len) : (at += 1) {
        for (0..ncls) |c| {
            moved.clearRetainingCapacity();
            const b = reps[c];
            for (sets.items[at]) |s| for (n.edges.items[s].items) |e| if (e.lo <= b and b <= e.hi) try moved.append(a, e.to);
            if (moved.items.len == 0) {
                try next.append(a, dead);
                continue;
            }
            try n.closure(a, moved.items, &scratch, seen);
            const id = try intern(a, &sets, &index, &accept, scratch.items, frag.end);
            if (sets.items.len > max_dfa) {
                diag.set("regex is too complex (over {d} DFA states)", .{max_dfa});
                return error.Unsupported;
            }
            try next.append(a, id);
        }
    }
    // keep only states that still reach a match
    const count = sets.items.len;
    const live = try a.alloc(bool, count);
    for (live, accept.items) |*l, acc| l.* = acc;
    var changed = true;
    while (changed) {
        changed = false;
        for (0..count) |s| if (!live[s]) for (next.items[s * ncls ..][0..ncls]) |t| if (t != dead and live[t]) {
            live[s] = true;
            changed = true;
            break;
        };
    }
    if (!live[0]) {
        diag.set("regex matches nothing", .{});
        return error.Unsupported;
    }
    for (next.items) |*t| if (t.* != dead and !live[t.*]) {
        t.* = dead;
    };
    return .{ .classes = classes, .ncls = ncls, .next = next.items, .accept = accept.items, .start = 0 };
}

fn intern(a: Allocator, sets: *std.ArrayList([]const u32), index: *std.StringHashMapUnmanaged(u32), accept: *std.ArrayList(bool), set: []const u32, final: u32) Error!u32 {
    const key = std.mem.sliceAsBytes(set);
    if (index.get(key)) |id| return id;
    const own = try a.dupe(u32, set);
    const id: u32 = @intCast(sets.items.len);
    try sets.append(a, own);
    try index.put(a, std.mem.sliceAsBytes(own), id);
    try accept.append(a, std.mem.indexOfScalar(u32, own, final) != null);
    return id;
}

/// Whether `d` matches all of `text`.
pub fn matches(d: *const Dfa, text: []const u8) bool {
    var s = d.start;
    for (text) |b| {
        s = d.step(s, b);
        if (s == dead) return false;
    }
    return d.accept[s];
}

test "full matches, classes, counts and UTF-8" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: Diag = .{};
    const phone = try compile(a, "[0-9]{3}-\\d{4}", .full, &diag);
    try std.testing.expect(matches(&phone, "555-1234"));
    try std.testing.expect(!matches(&phone, "555-123"));
    try std.testing.expect(!matches(&phone, "5555-1234"));
    // a prefix that cannot finish is dead at once
    try std.testing.expectEqual(dead, phone.step(phone.start, 'x'));
    const word = try compile(a, "(?:yes|no)( please)?", .full, &diag);
    try std.testing.expect(matches(&word, "no please"));
    try std.testing.expect(!matches(&word, "nope"));
    const any = try compile(a, "a.c", .full, &diag);
    try std.testing.expect(matches(&any, "a\u{e9}c"));
    try std.testing.expect(matches(&any, "a\u{1f600}c"));
    try std.testing.expect(!matches(&any, "a\nc"));
    try std.testing.expect(!matches(&any, "a\xffc"));
    const neg = try compile(a, "[^\"]+", .full, &diag);
    try std.testing.expect(matches(&neg, "caf\u{e9}"));
    try std.testing.expect(!matches(&neg, "a\"b"));
    const lit = try compile(a, "caf\u{e9}|\\u00e9t\u{e9}", .full, &diag);
    try std.testing.expect(matches(&lit, "caf\u{e9}"));
    try std.testing.expect(matches(&lit, "\u{e9}t\u{e9}"));
}

test "search patterns match anywhere unless anchored" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: Diag = .{};
    const open = try compile(a, "ab", .search, &diag);
    try std.testing.expect(matches(&open, "xxabyy"));
    try std.testing.expect(!matches(&open, "axb"));
    const anchored = try compile(a, "^[a-z]+$", .search, &diag);
    try std.testing.expect(matches(&anchored, "abc"));
    try std.testing.expect(!matches(&anchored, "abc1"));
    const dollar = try compile(a, "^a\\$", .search, &diag);
    try std.testing.expect(matches(&dollar, "a$xyz"));
}

test "outside the subset is refused with a message" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: Diag = .{};
    for ([_][]const u8{ "(a)\\1", "(?=a)", "\\bword", "[\u{e9}]", "a{3,2}", "(ab", "ab)", "*a", "a{2000}" }) |bad| {
        try std.testing.expectError(error.Unsupported, compile(a, bad, .full, &diag));
        try std.testing.expect(diag.text().len > 0);
    }
}
