//! A MessagePack reader for AMDGPU code-object metadata (the NT_AMDGPU_METADATA note): the whole format's value kinds,
//! decoded into a tree whose strings and binaries point into the input. Extension values are kept as raw bytes.

const std = @import("std");

pub const Value = union(enum) {
    nil,
    boolean: bool,
    uint: u64,
    int: i64,
    float: f64,
    str: []const u8,
    bin: []const u8,
    array: []const Value,
    map: []const Pair,
    ext: struct { kind: i8, data: []const u8 },

    pub const Pair = struct { key: Value, value: Value };

    /// The value of string key `name` in a map; null for a non-map or an absent key.
    pub fn get(v: Value, name: []const u8) ?Value {
        if (v != .map) return null;
        for (v.map) |p| if (p.key == .str and std.mem.eql(u8, p.key.str, name)) return p.value;
        return null;
    }

    /// A non-negative integer, whichever encoding carried it.
    pub fn unsigned(v: Value) ?u64 {
        return switch (v) {
            .uint => |x| x,
            .int => |x| if (x >= 0) @intCast(x) else null,
            else => null,
        };
    }

    pub fn string(v: Value) ?[]const u8 {
        return if (v == .str) v.str else null;
    }
};

pub const Error = error{ Truncated, Reserved, TooDeep } || std.mem.Allocator.Error;

pub fn decode(a: std.mem.Allocator, bytes: []const u8) Error!Value {
    var r: Reader = .{ .bytes = bytes };
    return r.value(a, 0);
}

const Reader = struct {
    bytes: []const u8,
    at: usize = 0,

    fn take(r: *Reader, n: usize) Error![]const u8 {
        if (n > r.bytes.len - r.at) return error.Truncated;
        const s = r.bytes[r.at..][0..n];
        r.at += n;
        return s;
    }

    fn int(r: *Reader, comptime T: type) Error!T {
        const s = try r.take(@sizeOf(T));
        return std.mem.readInt(T, s[0..@sizeOf(T)], .big);
    }

    fn value(r: *Reader, a: std.mem.Allocator, depth: u32) Error!Value {
        if (depth > 64) return error.TooDeep;
        const b = (try r.take(1))[0];
        return switch (b) {
            0x00...0x7f => .{ .uint = b },
            0x80...0x8f => .{ .map = try r.map(a, b & 0x0f, depth) },
            0x90...0x9f => .{ .array = try r.array(a, b & 0x0f, depth) },
            0xa0...0xbf => .{ .str = try r.take(b & 0x1f) },
            0xc0 => .nil,
            0xc1 => error.Reserved,
            0xc2 => .{ .boolean = false },
            0xc3 => .{ .boolean = true },
            0xc4 => .{ .bin = try r.take(try r.int(u8)) },
            0xc5 => .{ .bin = try r.take(try r.int(u16)) },
            0xc6 => .{ .bin = try r.take(try r.int(u32)) },
            0xc7 => r.ext(try r.int(u8)),
            0xc8 => r.ext(try r.int(u16)),
            0xc9 => r.ext(try r.int(u32)),
            0xca => .{ .float = @as(f32, @bitCast(try r.int(u32))) },
            0xcb => .{ .float = @as(f64, @bitCast(try r.int(u64))) },
            0xcc => .{ .uint = try r.int(u8) },
            0xcd => .{ .uint = try r.int(u16) },
            0xce => .{ .uint = try r.int(u32) },
            0xcf => .{ .uint = try r.int(u64) },
            0xd0 => .{ .int = try r.int(i8) },
            0xd1 => .{ .int = try r.int(i16) },
            0xd2 => .{ .int = try r.int(i32) },
            0xd3 => .{ .int = try r.int(i64) },
            0xd4 => r.ext(1),
            0xd5 => r.ext(2),
            0xd6 => r.ext(4),
            0xd7 => r.ext(8),
            0xd8 => r.ext(16),
            0xd9 => .{ .str = try r.take(try r.int(u8)) },
            0xda => .{ .str = try r.take(try r.int(u16)) },
            0xdb => .{ .str = try r.take(try r.int(u32)) },
            0xdc => .{ .array = try r.array(a, try r.int(u16), depth) },
            0xdd => .{ .array = try r.array(a, try r.int(u32), depth) },
            0xde => .{ .map = try r.map(a, try r.int(u16), depth) },
            0xdf => .{ .map = try r.map(a, try r.int(u32), depth) },
            0xe0...0xff => .{ .int = @as(i8, @bitCast(b)) },
        };
    }

    fn ext(r: *Reader, n: usize) Error!Value {
        const kind: i8 = @bitCast((try r.take(1))[0]);
        return .{ .ext = .{ .kind = kind, .data = try r.take(n) } };
    }

    fn array(r: *Reader, a: std.mem.Allocator, n: usize, depth: u32) Error![]const Value {
        if (n > r.bytes.len - r.at) return error.Truncated; // each element takes a byte at least
        const out = try a.alloc(Value, n);
        for (out) |*v| v.* = try r.value(a, depth + 1);
        return out;
    }

    fn map(r: *Reader, a: std.mem.Allocator, n: usize, depth: u32) Error![]const Value.Pair {
        if (n > (r.bytes.len - r.at) / 2) return error.Truncated;
        const out = try a.alloc(Value.Pair, n);
        for (out) |*p| p.* = .{ .key = try r.value(a, depth + 1), .value = try r.value(a, depth + 1) };
        return out;
    }
};

test "every value kind decodes, big-endian, into the input's bytes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // {"a": [1, -1, 300, -200, true, nil, 1.5], "bb": "xyz", "c": {"d": 70000}}
    const bytes = [_]u8{
        0x83,
        0xa1, 'a', 0x97, 0x01, 0xff, 0xcd, 0x01, 0x2c, 0xd1, 0xff, 0x38, 0xc3, 0xc0, 0xcb, 0x3f, 0xf8, 0, 0, 0, 0, 0, 0,
        0xa2, 'b', 'b', 0xd9, 3, 'x', 'y', 'z',
        0xa1, 'c', 0x81, 0xa1, 'd', 0xce, 0x00, 0x01, 0x11, 0x70,
    };
    const v = try decode(a, &bytes);
    const arr = v.get("a").?.array;
    try std.testing.expectEqual(@as(u64, 1), arr[0].unsigned().?);
    try std.testing.expectEqual(@as(i64, -1), arr[1].int);
    try std.testing.expectEqual(@as(u64, 300), arr[2].unsigned().?);
    try std.testing.expectEqual(@as(i64, -200), arr[3].int);
    try std.testing.expect(arr[4].boolean and arr[5] == .nil);
    try std.testing.expectEqual(@as(f64, 1.5), arr[6].float);
    try std.testing.expectEqualStrings("xyz", v.get("bb").?.string().?);
    try std.testing.expectEqual(@as(u64, 70000), v.get("c").?.get("d").?.unsigned().?);
    try std.testing.expect(v.get("zz") == null);
}

test "truncated and reserved inputs are refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.Truncated, decode(arena.allocator(), &.{ 0xa5, 'a' }));
    try std.testing.expectError(error.Truncated, decode(arena.allocator(), &.{ 0xdd, 0xff, 0xff, 0xff, 0xff }));
    try std.testing.expectError(error.Reserved, decode(arena.allocator(), &.{0xc1}));
}
