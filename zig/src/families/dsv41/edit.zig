//! TF_DS_EDIT's file: blocks' wo_b / shared w2 outputs edited to g * (y - alpha r (r . y)) (docs/UNCENSORED.md).
const std = @import("std");
const core = @import("core");
const st = core.safetensors;

pub const Which = enum {
    wo_b,
    shared_w2,

    fn prefix(w: Which) []const u8 {
        return switch (w) {
            .wo_b => "attn.wo_b",
            .shared_w2 => "ffn.shared_experts.w2",
        };
    }
};

/// One projection's edit as the file holds it (r and g: D floats each, the file's little-endian bytes).
pub const Vectors = struct { r: []const u8, g: []const u8, alpha: f32 };

pub const File = struct {
    f: st.File,
    /// The file's SHA-256 (hex): its first 16 characters name it in the handshake's plan (" edit=").
    sha256: [64]u8,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !File {
        var f = try st.File.open(gpa, io, path);
        errdefer f.close(io);
        var h: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(f.map.memory, &h, .{});
        return .{ .f = f, .sha256 = std.fmt.bytesToHex(h, .lower) };
    }

    pub fn close(e: *File, io: std.Io) void {
        e.f.close(io);
    }

    /// Every tensor is a target block's edit part, F32 of its shape, each projection's three together; else refused.
    pub fn check(e: *const File, layers: usize, d: usize) !void {
        var it = e.f.names.iterator();
        var parts: usize = 0;
        while (it.next()) |kv| {
            const name = kv.key_ptr.*;
            const p = parse(name) orelse return error.BadEditTensor;
            if (p.block >= layers) return error.BadEditTensor;
            parts += 1;
        }
        var whole: usize = 0;
        for (0..layers) |i| for ([_]Which{ .wo_b, .shared_w2 }) |w| {
            if (try e.get(i, w, d) != null) whole += 1;
        };
        if (parts != 3 * whole or whole == 0) return error.BadEditTensor;
    }

    /// Block `block`'s edit of `which`, null when the file leaves it as it is.
    pub fn get(e: *const File, block: usize, which: Which, d: usize) !?Vectors {
        var nb: [96]u8 = undefined;
        const r = e.f.get(try std.fmt.bufPrint(&nb, "layers.{d}.{s}.edit.r", .{ block, which.prefix() }));
        const g = e.f.get(try std.fmt.bufPrint(&nb, "layers.{d}.{s}.edit.g", .{ block, which.prefix() }));
        const a = e.f.get(try std.fmt.bufPrint(&nb, "layers.{d}.{s}.edit.alpha", .{ block, which.prefix() }));
        if (r == null and g == null and a == null) return null;
        const rt = r orelse return error.BadEditTensor;
        const gt = g orelse return error.BadEditTensor;
        const at = a orelse return error.BadEditTensor;
        if (!rt.is(.f32, &.{d}) or !gt.is(.f32, &.{d}) or !at.is(.f32, &.{1})) return error.BadEditTensor;
        const alpha: f32 = @bitCast(std.mem.readInt(u32, at.bytes[0..4], .little));
        if (!std.math.isFinite(alpha) or !finite(rt.bytes) or !finite(gt.bytes)) return error.BadEditTensor;
        return .{ .r = rt.bytes, .g = gt.bytes, .alpha = alpha };
    }
};

const Part = struct { block: usize, which: Which };

/// Every f32 of `bytes` (little-endian) is finite.
fn finite(bytes: []const u8) bool {
    var i: usize = 0;
    while (i + 4 <= bytes.len) : (i += 4) {
        const v: f32 = @bitCast(std.mem.readInt(u32, bytes[i..][0..4], .little));
        if (!std.math.isFinite(v)) return false;
    }
    return true;
}

/// `layers.{i}.{prefix}.edit.{r,g,alpha}` -> (i, which).
fn parse(name: []const u8) ?Part {
    if (!std.mem.startsWith(u8, name, "layers.")) return null;
    const rest = name["layers.".len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const block = std.fmt.parseInt(usize, rest[0..dot], 10) catch return null;
    const tail = rest[dot + 1 ..];
    for ([_]Which{ .wo_b, .shared_w2 }) |w| {
        if (!std.mem.startsWith(u8, tail, w.prefix())) continue;
        const part = tail[w.prefix().len..];
        for ([_][]const u8{ ".edit.r", ".edit.g", ".edit.alpha" }) |s| if (std.mem.eql(u8, part, s)) return .{ .block = block, .which = w };
    }
    return null;
}

/// A safetensors file at `path` of the given F32 tensors (name, values), for the tests.
fn writeFake(a: std.mem.Allocator, io: std.Io, path: []const u8, tensors: []const struct { []const u8, []const f32 }) !void {
    var hdr: std.ArrayList(u8) = .empty;
    try hdr.append(a, '{');
    var off: usize = 0;
    for (tensors, 0..) |t, i| {
        const n = t[1].len;
        try hdr.appendSlice(a, try std.fmt.allocPrint(a, "{s}\"{s}\": {{\"dtype\": \"F32\", \"shape\": [{d}], \"data_offsets\": [{d}, {d}]}}", .{ if (i == 0) "" else ", ", t[0], n, off, off + 4 * n }));
        off += 4 * n;
    }
    try hdr.append(a, '}');
    var out: std.ArrayList(u8) = .empty;
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, hdr.items.len, .little);
    try out.appendSlice(a, &len);
    try out.appendSlice(a, hdr.items);
    for (tensors) |t| try out.appendSlice(a, std.mem.sliceAsBytes(t[1]));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.items });
}

test "an edit file: each projection's r, g and alpha by block; strays and partial edits refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const dir = try std.fmt.allocPrint(a, ".tf-edit-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    try std.Io.Dir.cwd().createDirPath(io, dir);
    const r = [_]f32{ 0.6, 0.8, 0, 0 };
    const g = [_]f32{ 1, 0.5, 1, 1 };
    const ok = try std.fs.path.join(a, &.{ dir, "ok.safetensors" });
    try writeFake(a, io, ok, &.{
        .{ "layers.2.attn.wo_b.edit.r", &r },
        .{ "layers.2.attn.wo_b.edit.g", &g },
        .{ "layers.2.attn.wo_b.edit.alpha", &.{6.0} },
    });
    var e = try File.open(std.testing.allocator, io, ok);
    defer e.close(io);
    try e.check(4, 4);
    const v = (try e.get(2, .wo_b, 4)).?;
    try std.testing.expectEqual(@as(f32, 6.0), v.alpha);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&g), v.g);
    try std.testing.expect(try e.get(2, .shared_w2, 4) == null);
    try std.testing.expect(try e.get(3, .wo_b, 4) == null);
    try std.testing.expectError(error.BadEditTensor, e.get(2, .wo_b, 8)); // another hidden size
    try std.testing.expectError(error.BadEditTensor, e.check(2, 4)); // block 2 past the target's layers

    const partial = try std.fs.path.join(a, &.{ dir, "partial.safetensors" });
    try writeFake(a, io, partial, &.{
        .{ "layers.1.ffn.shared_experts.w2.edit.r", &r },
        .{ "layers.1.ffn.shared_experts.w2.edit.alpha", &.{6.0} },
    });
    var p = try File.open(std.testing.allocator, io, partial);
    defer p.close(io);
    try std.testing.expectError(error.BadEditTensor, p.check(4, 4));

    const stray = try std.fs.path.join(a, &.{ dir, "stray.safetensors" });
    try writeFake(a, io, stray, &.{
        .{ "layers.1.attn.wo_b.edit.r", &r },
        .{ "layers.1.attn.wo_b.edit.g", &g },
        .{ "layers.1.attn.wo_b.edit.alpha", &.{6.0} },
        .{ "mtp.0.attn.wo_b.edit.r", &r },
    });
    var t = try File.open(std.testing.allocator, io, stray);
    defer t.close(io);
    try std.testing.expectError(error.BadEditTensor, t.check(4, 4));

    const nan = try std.fs.path.join(a, &.{ dir, "nan.safetensors" });
    try writeFake(a, io, nan, &.{
        .{ "layers.1.attn.wo_b.edit.r", &r },
        .{ "layers.1.attn.wo_b.edit.g", &.{ 1, std.math.nan(f32), 1, 1 } },
        .{ "layers.1.attn.wo_b.edit.alpha", &.{6.0} },
    });
    var n = try File.open(std.testing.allocator, io, nan);
    defer n.close(io);
    try std.testing.expectError(error.BadEditTensor, n.get(1, .wo_b, 4)); // a scale that is not a number
}

test "edit tensor names" {
    try std.testing.expectEqual(Part{ .block = 36, .which = .shared_w2 }, parse("layers.36.ffn.shared_experts.w2.edit.alpha").?);
    try std.testing.expectEqual(Part{ .block = 10, .which = .wo_b }, parse("layers.10.attn.wo_b.edit.r").?);
    try std.testing.expect(parse("layers.10.attn.wo_b.weight") == null);
    try std.testing.expect(parse("layers.x.attn.wo_b.edit.r") == null);
    try std.testing.expect(parse("mtp.0.attn.wo_b.edit.r") == null);
}
