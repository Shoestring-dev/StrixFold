//! A kept prompt state on the wire (POST /cache/export, /cache/import): a signed header over sf-disk's payload layout.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Hmac = std.crypto.auth.hmac.sha2.HmacSha256;

pub const version: u32 = 1;
const magic = "SFCX".*;
/// The header: magic, version, namespace, positions, ids, blocks, the ids' SHA-256, slot bytes, block bytes.
pub const head_bytes = 84;
pub const mac_bytes = 32;
pub const Key = [32]u8;

/// What an engine's states look like: the identity (sf-disk's namespace) and sizes every body is checked against.
pub const Shape = struct {
    ns: [16]u8,
    block_tokens: u32, // a block's positions
    max_blocks: u32, // the most blocks a state lists
    max_positions: u32, // the most positions a state covers (the context window)
    lookahead: u32 = 0, // ids past the positions a state carries (prompt_cache Rules.lookahead)
    slot_bytes: u64, // the state's own bytes
    block_bytes: u64, // one block's rows, every plane

    /// The longest body this engine takes: a state at max_positions, its ids, its blocks, the MAC.
    pub fn bodyMax(s: Shape) error{LengthOverflow}!u64 {
        const k = @min(s.max_positions / s.block_tokens, s.max_blocks);
        const ids = try mul(4, @as(u64, s.max_positions) + s.lookahead);
        return try add(try add(try add(head_bytes + mac_bytes, ids), s.slot_bytes), try mul(k, s.block_bytes));
    }
};

pub const Error = error{
    ShortBody,
    BadMagic,
    BadVersion,
    OtherIdentity,
    BadPositions,
    BadIdCount,
    BadBlockCount,
    BadSlotLength,
    BadBlockLength,
    LengthOverflow,
    TrailingBytes,
    BadIdHash,
};

/// The field each refusal names (the route's 400).
pub fn message(e: Error) []const u8 {
    return switch (e) {
        error.ShortBody => "body: shorter than its header announces",
        error.BadMagic => "magic: not a prompt state this server exports",
        error.BadVersion => "version: not this server's",
        error.OtherIdentity => "identity: the state is another engine's (build, checkpoint, plan or settings)",
        error.BadPositions => "positions: 0, past the context window, or not a whole number of blocks",
        error.BadIdCount => "ids: not the positions plus the engine's lookahead",
        error.BadBlockCount => "blocks: not the positions' block count",
        error.BadSlotLength => "slot_bytes: not this engine's state size",
        error.BadBlockLength => "block_bytes: not this engine's block size",
        error.LengthOverflow => "lengths: their sum overflows",
        error.TrailingBytes => "body: longer than its header announces",
        error.BadIdHash => "ids: their hash is not the header's",
    };
}

/// A body's parts, slices into it (nothing copied): the positions, the ids' little-endian bytes, the slot, the blocks.
pub const Parsed = struct {
    positions: u32,
    blocks: u32,
    ids: []const u8,
    slot: []const u8,
    rows: []const u8,

    /// The ids as numbers, into `out` (its length the id count).
    pub fn idsInto(p: Parsed, out: []u32) void {
        for (out, 0..) |*t, i| t.* = std.mem.readInt(u32, p.ids[4 * i ..][0..4], .little);
    }

    pub fn idCount(p: Parsed) usize {
        return p.ids.len / 4;
    }
};

fn add(a: u64, b: u64) error{LengthOverflow}!u64 {
    return std.math.add(u64, a, b) catch error.LengthOverflow;
}

fn mul(a: u64, b: u64) error{LengthOverflow}!u64 {
    return std.math.mul(u64, a, b) catch error.LengthOverflow;
}

/// Every header field checked against `s` and the body's length before anything is read past the header.
pub fn parse(s: Shape, body: []const u8) Error!Parsed {
    if (body.len < head_bytes + mac_bytes) return error.ShortBody;
    const h = body[0..head_bytes];
    if (!std.mem.eql(u8, h[0..4], &magic)) return error.BadMagic;
    if (std.mem.readInt(u32, h[4..8], .little) != version) return error.BadVersion;
    if (!std.mem.eql(u8, h[8..24], &s.ns)) return error.OtherIdentity;
    const positions = std.mem.readInt(u32, h[24..28], .little);
    const ids_n = std.mem.readInt(u32, h[28..32], .little);
    const blocks = std.mem.readInt(u32, h[32..36], .little);
    const slot_bytes = std.mem.readInt(u64, h[68..76], .little);
    const block_bytes = std.mem.readInt(u64, h[76..84], .little);
    if (positions == 0 or positions > s.max_positions or positions % s.block_tokens != 0) return error.BadPositions;
    if (@as(u64, ids_n) != @as(u64, positions) + s.lookahead) return error.BadIdCount;
    if (blocks != positions / s.block_tokens or blocks > s.max_blocks) return error.BadBlockCount;
    if (slot_bytes != s.slot_bytes) return error.BadSlotLength;
    if (block_bytes != s.block_bytes) return error.BadBlockLength;
    const ids_len = try mul(4, ids_n);
    const rows_len = try mul(blocks, block_bytes);
    const need = try add(try add(try add(head_bytes + mac_bytes, ids_len), slot_bytes), rows_len);
    if (body.len < need) return error.ShortBody;
    if (body.len > need) return error.TrailingBytes;
    const ids = body[head_bytes..][0..ids_len];
    var d: [32]u8 = undefined;
    Sha256.hash(ids, &d, .{});
    if (!std.mem.eql(u8, &d, h[36..68])) return error.BadIdHash;
    const slot = body[head_bytes + ids_len ..][0..slot_bytes];
    return .{ .positions = positions, .blocks = blocks, .ids = ids, .slot = slot, .rows = body[head_bytes + ids_len + slot_bytes ..][0..rows_len] };
}

/// The header of a state of `positions` positions carrying `ids` (their little-endian bytes), under `s`.
pub fn encodeHead(s: Shape, positions: u32, ids: []const u8) [head_bytes]u8 {
    var h: [head_bytes]u8 = undefined;
    h[0..4].* = magic;
    std.mem.writeInt(u32, h[4..8], version, .little);
    h[8..24].* = s.ns;
    std.mem.writeInt(u32, h[24..28], positions, .little);
    std.mem.writeInt(u32, h[28..32], @intCast(ids.len / 4), .little);
    std.mem.writeInt(u32, h[32..36], positions / s.block_tokens, .little);
    Sha256.hash(ids, h[36..68], .{});
    std.mem.writeInt(u64, h[68..76], s.slot_bytes, .little);
    std.mem.writeInt(u64, h[76..84], s.block_bytes, .little);
    return h;
}

/// HMAC-SHA256 under `key` over `parts` in order (the header, the ids, the slot, the blocks): the body's last 32 bytes.
pub fn sign(key: *const Key, parts: []const []const u8) [mac_bytes]u8 {
    var m = Hmac.init(key);
    for (parts) |p| m.update(p);
    var out: [mac_bytes]u8 = undefined;
    m.final(&out);
    return out;
}

/// Whether `body` ends in the MAC of the rest under `key`, compared in constant time; false for a body too short.
pub fn verify(key: *const Key, body: []const u8) bool {
    if (body.len < head_bytes + mac_bytes) return false;
    const want = sign(key, &.{body[0 .. body.len - mac_bytes]});
    return std.crypto.timing_safe.eql([mac_bytes]u8, want, body[body.len - mac_bytes ..][0..mac_bytes].*);
}

/// The ids as little-endian bytes (the body's ids part), in `out` (4 bytes an id).
pub fn idBytes(ids: []const u32, out: []u8) void {
    for (ids, 0..) |t, i| std.mem.writeInt(u32, out[4 * i ..][0..4], t, .little);
}

test {
    _ = @import("cache_wire_test.zig");
}
