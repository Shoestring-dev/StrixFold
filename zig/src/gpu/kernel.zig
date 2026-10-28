//! A kernel's ABI as a Zig type (ADAPTER.md 6.2): `Params` is an extern struct with one field per C parameter, in
//! order; device addresses are u64; a by-value C struct is one field of its translated type (zig/kernels/hip/dsv41/
//! abi.h). The parameter table (offset, size, kind of each parameter in the kernarg segment) is computed here at compile
//! time and compared with the code object's own metadata when the kernel is bound, so a declaration that disagrees
//! with the compiled kernel never launches. Packing is the same sequence of Args.add calls the family writes by hand.

const std = @import("std");

pub const Kind = enum { address_or_u64, scalar, by_value_struct };
pub const Param = struct { name: []const u8, offset: u32, size: u32, kind: Kind };

/// The kernarg layout of `P`'s fields as C parameters (each at its own alignment, in order).
pub fn table(comptime P: type) [@typeInfo(P).@"struct".field_names.len]Param {
    const info = @typeInfo(P).@"struct";
    if (info.layout != .@"extern") @compileError(@typeName(P) ++ ": kernel parameters must be an extern struct");
    var out: [info.field_names.len]Param = undefined;
    var at: u32 = 0;
    inline for (info.field_names, info.field_types, 0..) |name, T, i| {
        const kind: Kind = switch (@typeInfo(T)) {
            .@"struct" => |s| if (s.layout == .@"extern") .by_value_struct else @compileError(name ++ ": by-value structs must be extern"),
            .int => if (T == u64) .address_or_u64 else .scalar,
            .float => .scalar,
            else => @compileError(name ++ ": kernel parameters are fixed-width ints, floats, u64 addresses or extern structs"),
        };
        at = std.mem.alignForward(u32, at, @alignOf(T));
        out[i] = .{ .name = name, .offset = at, .size = @sizeOf(T), .kind = kind };
        at += @sizeOf(T);
    }
    return out;
}

/// The first disagreement between a declared table and a kernel's explicit metadata arguments (offset, size,
/// value_kind), as text in `buf`; null when they agree.
pub fn mismatch(params: []const Param, args: anytype, buf: []u8) ?[]const u8 {
    if (params.len != args.len) return std.fmt.bufPrint(buf, "{d} parameters declared, {d} in the code object", .{ params.len, args.len }) catch "parameter count";
    for (params, args) |p, a| {
        if (p.offset != a.offset or p.size != a.size)
            return std.fmt.bufPrint(buf, "{s}: declared at {d} size {d}, compiled at {d} size {d}", .{ p.name, p.offset, p.size, a.offset, a.size }) catch "layout";
        const ok = switch (p.kind) {
            .address_or_u64 => std.mem.eql(u8, a.kind, "global_buffer") or std.mem.eql(u8, a.kind, "by_value"),
            .scalar, .by_value_struct => std.mem.eql(u8, a.kind, "by_value"),
        };
        if (!ok) return std.fmt.bufPrint(buf, "{s}: declared {s}, compiled as {s}", .{ p.name, @tagName(p.kind), a.kind }) catch "kind";
    }
    return null;
}

/// Packs `p` into `args` field by field (the order and C alignment `table(P)` describes).
pub fn pack(comptime P: type, p: P, args: anytype) void {
    inline for (@typeInfo(P).@"struct".field_names) |name| args.add(@field(p, name));
}

test "tables follow C parameter alignment and classify fields" {
    const S = extern struct { a: u64, b: i32 };
    const P = extern struct { x: u8, addr: u64, h: u16, f: f32, s: S, out: u64 };
    const t = comptime table(P);
    try std.testing.expectEqual(@as(u32, 0), t[0].offset);
    try std.testing.expectEqual(@as(u32, 8), t[1].offset);
    try std.testing.expectEqual(@as(u32, 16), t[2].offset);
    try std.testing.expectEqual(@as(u32, 20), t[3].offset);
    try std.testing.expectEqual(@as(u32, 24), t[4].offset);
    try std.testing.expectEqual(@as(u32, 16), t[4].size);
    try std.testing.expectEqual(@as(u32, 40), t[5].offset);
    try std.testing.expectEqual(Kind.by_value_struct, t[4].kind);
    try std.testing.expectEqual(Kind.address_or_u64, t[1].kind);
    try std.testing.expectEqual(Kind.scalar, t[0].kind);
}

test "a declaration that disagrees with the metadata is named" {
    const A = struct { offset: u32, size: u32, kind: []const u8 };
    const P = extern struct { in: u64, out: u64, n: u32 };
    const t = comptime table(P);
    var buf: [128]u8 = undefined;
    const good = [_]A{ .{ .offset = 0, .size = 8, .kind = "global_buffer" }, .{ .offset = 8, .size = 8, .kind = "global_buffer" }, .{ .offset = 16, .size = 4, .kind = "by_value" } };
    try std.testing.expect(mismatch(&t, &good, &buf) == null);
    const wide = [_]A{ good[0], good[1], .{ .offset = 16, .size = 8, .kind = "by_value" } };
    try std.testing.expectEqualStrings("n: declared at 16 size 4, compiled at 16 size 8", mismatch(&t, &wide, &buf).?);
    try std.testing.expect(mismatch(&t, good[0..2], &buf) != null);
    const ptr_scalar = [_]A{ good[0], good[1], .{ .offset = 16, .size = 4, .kind = "global_buffer" } };
    try std.testing.expect(mismatch(&t, &ptr_scalar, &buf) != null);
}
