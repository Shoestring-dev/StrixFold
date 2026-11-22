//! cache_wire's tests: a body built as export builds one, then every malformed body import must refuse by name.
const std = @import("std");
const wire = @import("cache_wire.zig");
const Shape = wire.Shape;

const shape: Shape = .{ .ns = @splat(7), .block_tokens = 4, .max_blocks = 8, .max_positions = 32, .lookahead = 1, .slot_bytes = 5, .block_bytes = 6 };
const key: wire.Key = @splat(0x4b);

/// A signed body of a state at `positions` under `s`, the parts' bytes counted up from 1.
fn body(a: std.mem.Allocator, s: Shape, positions: u32) ![]u8 {
    const ids_n = positions + s.lookahead;
    const ids = try a.alloc(u32, ids_n);
    defer a.free(ids);
    for (ids, 0..) |*t, i| t.* = @intCast(1000 + i);
    const ids_b = try a.alloc(u8, 4 * ids_n);
    defer a.free(ids_b);
    wire.idBytes(ids, ids_b);
    const head = wire.encodeHead(s, positions, ids_b);
    const slot = try a.alloc(u8, s.slot_bytes);
    defer a.free(slot);
    for (slot, 0..) |*b, i| b.* = @intCast(1 + i);
    const rows = try a.alloc(u8, s.block_bytes * (positions / s.block_tokens));
    defer a.free(rows);
    for (rows, 0..) |*b, i| b.* = @intCast(100 + i % 100);
    const mac = wire.sign(&key, &.{ &head, ids_b, slot, rows });
    return std.mem.concat(a, u8, &.{ &head, ids_b, slot, rows, &mac });
}

test "a body reads back into its parts, and its MAC holds only under its key and unchanged" {
    const a = std.testing.allocator;
    const b = try body(a, shape, 8);
    defer a.free(b);
    try std.testing.expectEqual(wire.head_bytes + 4 * 9 + 5 + 2 * 6 + wire.mac_bytes, b.len);
    try std.testing.expect(wire.verify(&key, b));
    const p = try wire.parse(shape, b);
    try std.testing.expectEqual(@as(u32, 8), p.positions);
    try std.testing.expectEqual(@as(u32, 2), p.blocks);
    try std.testing.expectEqual(@as(usize, 9), p.idCount());
    var ids: [9]u32 = undefined;
    p.idsInto(&ids);
    try std.testing.expectEqual(@as(u32, 1000), ids[0]);
    try std.testing.expectEqual(@as(u32, 1008), ids[8]);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5 }, p.slot);
    try std.testing.expectEqual(@as(usize, 12), p.rows.len);
    try std.testing.expectEqual(@as(u8, 100), p.rows[0]);
    // another key, a flipped payload byte, a flipped MAC byte, a body shorter than header plus MAC
    var other = key;
    other[0] ^= 1;
    try std.testing.expect(!wire.verify(&other, b));
    b[wire.head_bytes + 40] ^= 1;
    try std.testing.expect(!wire.verify(&key, b));
    b[wire.head_bytes + 40] ^= 1;
    b[b.len - 1] ^= 1;
    try std.testing.expect(!wire.verify(&key, b));
    try std.testing.expect(!wire.verify(&key, b[0 .. wire.head_bytes + 10]));
    try std.testing.expectEqual(@as(u64, wire.head_bytes + wire.mac_bytes + 4 * 33 + 5 + 8 * 6), try shape.bodyMax());
}

fn expectRefused(e: wire.Error, b: []const u8) !void {
    try std.testing.expectError(e, wire.parse(shape, b));
    try std.testing.expect(wire.message(e).len > 0);
}

test "malformed bodies are refused by the field at fault before any part is read" {
    const a = std.testing.allocator;
    const good = try body(a, shape, 8);
    defer a.free(good);
    const b = try a.dupe(u8, good);
    defer a.free(b);
    // truncated: inside the header, inside a part, and one byte short
    try expectRefused(error.ShortBody, b[0..10]);
    try expectRefused(error.ShortBody, b[0 .. wire.head_bytes + 20]);
    try expectRefused(error.ShortBody, b[0 .. b.len - 1]);
    // the magic of sf-disk's files (another format), another version
    b[0..4].* = "SFDS".*;
    try expectRefused(error.BadMagic, b);
    @memcpy(b, good);
    std.mem.writeInt(u32, b[4..8], 2, .little);
    try expectRefused(error.BadVersion, b);
    // another engine's identity
    @memcpy(b, good);
    b[8] ^= 1;
    try expectRefused(error.OtherIdentity, b);
    // oversized counts: positions and blocks at u32 max, positions off a block end, past the window
    @memcpy(b, good);
    std.mem.writeInt(u32, b[24..28], std.math.maxInt(u32), .little);
    try expectRefused(error.BadPositions, b);
    @memcpy(b, good);
    std.mem.writeInt(u32, b[24..28], 7, .little);
    try expectRefused(error.BadPositions, b);
    @memcpy(b, good);
    std.mem.writeInt(u32, b[24..28], 36, .little);
    try expectRefused(error.BadPositions, b);
    @memcpy(b, good);
    std.mem.writeInt(u32, b[32..36], std.math.maxInt(u32), .little);
    try expectRefused(error.BadBlockCount, b);
    @memcpy(b, good);
    std.mem.writeInt(u32, b[28..32], 8, .little);
    try expectRefused(error.BadIdCount, b);
    // part lengths other than the engine's (u64 max among them: no multiply happens on them)
    @memcpy(b, good);
    std.mem.writeInt(u64, b[68..76], std.math.maxInt(u64), .little);
    try expectRefused(error.BadSlotLength, b);
    @memcpy(b, good);
    std.mem.writeInt(u64, b[76..84], std.math.maxInt(u64), .little);
    try expectRefused(error.BadBlockLength, b);
    // trailing bytes, and ids that are not the header's hash
    const longer = try std.mem.concat(a, u8, &.{ good, "x" });
    defer a.free(longer);
    try expectRefused(error.TrailingBytes, longer);
    @memcpy(b, good);
    b[wire.head_bytes] ^= 1;
    try expectRefused(error.BadIdHash, b);
}

test "a shape whose lengths overflow is refused as such, by bodyMax and by parse" {
    var huge = shape;
    huge.block_bytes = 1 << 62;
    try std.testing.expectError(error.LengthOverflow, huge.bodyMax());
    const zeros: [4 * 33]u8 = @splat(0);
    var h = wire.encodeHead(huge, 32, &zeros);
    var b: [wire.head_bytes + wire.mac_bytes]u8 = undefined;
    @memcpy(b[0..wire.head_bytes], &h);
    try std.testing.expectError(error.LengthOverflow, wire.parse(huge, &b));
    h = wire.encodeHead(shape, 4, zeros[0 .. 4 * 5]);
    @memcpy(b[0..wire.head_bytes], &h);
    try std.testing.expectError(error.ShortBody, wire.parse(shape, &b)); // (the lengths fit; the parts are missing)
}
