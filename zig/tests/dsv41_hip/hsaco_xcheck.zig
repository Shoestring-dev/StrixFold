//! hsaco.zig against an independent reader: every kernel of real code objects (the smoke and ABI-echo objects built by
//! this build) as llvm-readelf --notes prints its metadata, field by field. Host test, no GPU.

const std = @import("std");
const gpu = @import("gpu");
const objects = @import("xcheck_objects");

const Ref = struct {
    name: []const u8 = "",
    symbol: []const u8 = "",
    kernarg: u32 = 0,
    wave: u32 = 0,
    max_flat: u32 = 0,
    lds: u32 = 0,
    scratch: u32 = 0,
    sgpr: u32 = 0,
    vgpr: u32 = 0,
    args: std.ArrayList(gpu.hsaco.Arg) = .empty,
};

/// The amdhsa.kernels list of llvm-readelf's YAML rendering: kernel items at indent 2, their keys at 4, argument items
/// at 6 with keys at 8.
fn readelfKernels(a: std.mem.Allocator, text: []const u8) ![]Ref {
    var out: std.ArrayList(Ref) = .empty;
    var in_kernels = false;
    var in_args = false;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw| {
        var indent: usize = 0;
        while (indent < raw.len and raw[indent] == ' ') indent += 1;
        var content = raw[indent..];
        if (content.len == 0) continue;
        if (indent == 0) {
            in_kernels = std.mem.startsWith(u8, content, "amdhsa.kernels:");
            continue;
        }
        if (!in_kernels) continue;
        var item = false;
        if (std.mem.startsWith(u8, content, "- ")) {
            item = true;
            content = content[2..];
            indent += 2;
        }
        if (item and indent == 4) try out.append(a, .{});
        if (out.items.len == 0) continue;
        const k = &out.items[out.items.len - 1];
        const colon = std.mem.indexOfScalar(u8, content, ':');
        const key = if (colon) |c| content[0..c] else "";
        // only the .args list holds arguments (.language_version is a list at the same depth)
        if (indent == 4) in_args = std.mem.eql(u8, key, ".args");
        if (item and indent == 8 and in_args) try k.args.append(a, .{ .offset = 0, .size = 0, .kind = "" });
        if (colon == null) continue;
        const val = std.mem.trim(u8, content[colon.? + 1 ..], " '\"");
        if (indent == 8 and in_args and k.args.items.len > 0) {
            const arg = &k.args.items[k.args.items.len - 1];
            if (std.mem.eql(u8, key, ".offset")) arg.offset = try std.fmt.parseInt(u32, val, 10);
            if (std.mem.eql(u8, key, ".size")) arg.size = try std.fmt.parseInt(u32, val, 10);
            if (std.mem.eql(u8, key, ".value_kind")) arg.kind = val;
            continue;
        }
        if (indent != 4) continue;
        const fields = .{ .{ ".kernarg_segment_size", "kernarg" }, .{ ".wavefront_size", "wave" }, .{ ".max_flat_workgroup_size", "max_flat" }, .{ ".group_segment_fixed_size", "lds" }, .{ ".private_segment_fixed_size", "scratch" }, .{ ".sgpr_count", "sgpr" }, .{ ".vgpr_count", "vgpr" } };
        inline for (fields) |f| if (std.mem.eql(u8, key, f[0])) {
            @field(k, f[1]) = try std.fmt.parseInt(u32, val, 10);
        };
        if (std.mem.eql(u8, key, ".name")) k.name = val;
        if (std.mem.eql(u8, key, ".symbol")) k.symbol = val;
    }
    return out.items;
}

fn crossCheck(image: []const u8, readelf: []const u8) !usize {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ours = try gpu.hsaco.parse(a, image);
    const refs = try readelfKernels(a, readelf);
    try std.testing.expectEqual(refs.len, ours.kernels.len);
    try std.testing.expect(refs.len > 0);
    try std.testing.expectEqual(@as(u32, 0x4a), ours.mach());
    for (refs) |r| {
        const k = ours.find(r.name) orelse return error.KernelMissing;
        try std.testing.expectEqualStrings(r.symbol, k.symbol);
        try std.testing.expectEqual(r.kernarg, k.kernarg_size);
        try std.testing.expectEqual(r.wave, k.wave);
        try std.testing.expectEqual(r.max_flat, k.max_flat_workgroup_size);
        try std.testing.expectEqual(r.lds, k.lds_static);
        try std.testing.expectEqual(r.scratch, k.scratch);
        try std.testing.expectEqual(r.sgpr, k.sgpr);
        try std.testing.expectEqual(r.vgpr, k.vgpr);
        try std.testing.expectEqual(r.args.items.len, k.args.len);
        for (r.args.items, k.args) |x, y| {
            try std.testing.expectEqual(x.offset, y.offset);
            try std.testing.expectEqual(x.size, y.size);
            try std.testing.expectEqualStrings(x.kind, y.kind);
        }
    }
    return refs.len;
}

test "hsaco.zig reads the smoke kernels' metadata as llvm-readelf does" {
    try std.testing.expectEqual(@as(usize, 7), try crossCheck(&objects.smoke, objects.smoke_readelf));
}

test "hsaco.zig reads the ABI echo kernels' metadata as llvm-readelf does" {
    try std.testing.expectEqual(@as(usize, 8), try crossCheck(&objects.echo, objects.echo_readelf));
}

test "the echo kernels' explicit arguments are what their C declarations give" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const co = try gpu.hsaco.parse(arena.allocator(), &objects.echo);
    const p = co.find("tf_ds_abi_echo_params").?.explicit();
    const want = [_][3]u32{ .{ 0, 1, 0 }, .{ 8, 8, 0 }, .{ 16, 2, 0 }, .{ 20, 4, 0 }, .{ 24, 1, 0 }, .{ 32, 8, 0 }, .{ 40, 4, 0 }, .{ 48, 8, 0 } };
    try std.testing.expectEqual(want.len, p.len);
    for (want, p) |w, x| {
        try std.testing.expectEqual(w[0], x.offset);
        try std.testing.expectEqual(w[1], x.size);
    }
    const m = co.find("tf_ds_abi_echo_mixed").?.explicit();
    try std.testing.expectEqual(@as(usize, 2), m.len);
    try std.testing.expectEqual(@as(u32, 64), m[0].size);
    try std.testing.expectEqualStrings("by_value", m[0].kind);
    try std.testing.expectEqualStrings("global_buffer", m[1].kind);
    try std.testing.expectEqual(@as(u32, 1424), co.find("tf_ds_abi_echo_big").?.explicit()[0].size);
    // w-exl3's EXL3 echoes: glinear's 1,424-byte and rot_many's 392-byte structs; grouped_cp's list, its 144-byte
    // epilogue at 104 after ten addresses and six ints
    try std.testing.expectEqual(@as(u32, 1424), co.find("tf_ds_abi_echo_exl3_gargs").?.explicit()[0].size);
    try std.testing.expectEqual(@as(u32, 392), co.find("tf_ds_abi_echo_exl3_rargs").?.explicit()[0].size);
    const cp = co.find("tf_ds_abi_echo_exl3_cp").?.explicit();
    try std.testing.expectEqual(@as(usize, 18), cp.len);
    try std.testing.expectEqual(@as(u32, 104), cp[16].offset);
    try std.testing.expectEqual(@as(u32, 144), cp[16].size);
    try std.testing.expectEqual(@as(u32, 248), cp[17].offset);
    // the Zig declaration of the same kernel computes the same table
    const P = extern struct { a: u8, b: u64, c: u16, d: f32, e: i8, f: f64, g: i32, out: u64 };
    var buf: [128]u8 = undefined;
    try std.testing.expect(gpu.kernelTable.mismatch(&gpu.kernelTable.table(P), p, &buf) == null);
}
