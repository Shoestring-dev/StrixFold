//! The prompt path's EXL3 linear (exl3/prefill.py matmul, more than 128 rows): the input rows rotated into fp16
//! (rot_in: fp16(((x * suh) H) / sqrt(128))), the weight's trellis decoded into fp16 W_q [K, N] (unpack), then the
//! Triton GEMM (tri_markov.matmul) with the 128 x 128 Hadamard and svh. On HIP with the kit's exl3.gemm, the rotated
//! rows go into one GEMM that decodes the trellis itself (no W_q written). rot_in and unpack are the served build's
//! (linear.cu's cubin of tensorfold_exl3_linear_v7), launched as exl3_rot_in_cuda and exl3_unpack_cuda launch them.
const std = @import("std");
const gpu = @import("gpu");
const elf = @import("elf.zig");
const exl3_linear = @import("exl3_linear.zig");
const tri = @import("tri.zig");
const tri_markov = @import("tri_markov.zig");
const weights = @import("weights.zig");

pub const DType = exl3_linear.DType;

pub const Kernels = struct {
    module: gpu.Module,
    rot_in: gpu.Function,
    unpack: [17][3]?gpu.Function = @splat(@splat(null)),
    /// The AMD kit's exl3.gemm by {k2, bm 128 | 256} (hip_load.zig): the trellis decoded inside the GEMM, in place of
    /// unpack + the Triton _gemm; null in a CUDA build or a kit without it.
    gemm: [17][2]?gpu.Function = @splat(@splat(null)),
    /// The functions are the AMD kit's (hip_load.zig): unload leaves its modules alone.
    kit_owned: bool = false,

    /// linear.cu's cubin (the one with unpack_kernel): kernels found by name, since the anonymous namespace's hash
    /// changes with every build.
    pub fn load(d: *const gpu.Driver, cubin: []const u8) !Kernels {
        var k: Kernels = .{ .module = try gpu.Module.load(d, cubin), .rot_in = undefined };
        errdefer k.module.unload();
        var names = elf.Symbols.init(cubin) orelse return error.BadCubin;
        var found = false;
        while (names.next()) |name| {
            var buf: [256]u8 = undefined;
            if (std.mem.indexOf(u8, name, "rot_in_kernel") != null) {
                k.rot_in = try k.module.function(try std.mem.printSentinel(&buf, "{s}", .{name}, 0));
                found = true;
                continue;
            }
            const at = std.mem.indexOf(u8, name, "unpack_kernelILi") orelse continue;
            const t = template2(name[at + "unpack_kernel".len ..]) orelse continue;
            if (t[0] > 16 or t[1] > 2) continue;
            k.unpack[t[0]][t[1]] = try k.module.function(try std.mem.printSentinel(&buf, "{s}", .{name}, 0));
        }
        if (!found) return error.MissingKernel;
        return k;
    }

    pub fn unload(k: *Kernels) void {
        if (!k.kit_owned) k.module.unload();
        k.* = undefined;
    }
};

/// "ILi7ELi2EEEv..." -> {7, 2}
fn template2(s: []const u8) ?[2]u32 {
    var out: [2]u32 = undefined;
    var rest = s;
    for (&out) |*v| {
        const i = std.mem.indexOf(u8, rest, "Li") orelse return null;
        rest = rest[i + 2 ..];
        const e = std.mem.indexOfScalar(u8, rest, 'E') orelse return null;
        v.* = std.fmt.parseInt(u32, rest[0..e], 10) catch return null;
        rest = rest[e + 1 ..];
    }
    return out;
}

/// rot_in: xh [M, K] fp16 (contiguous) from x's M rows (row stride ldx elements, rows 16-byte aligned), K a multiple
/// of 128: a warp a 128-column block, four blocks a program, a row a grid row.
pub fn rotIn(k: *const Kernels, stream: gpu.Stream, x: u64, x_dtype: DType, ldx: usize, suh: u64, xh: u64, m: usize, kdim: usize) !void {
    if (kdim % 128 != 0 or x % 16 != 0 or (ldx * dtypeBytes(x_dtype)) % 16 != 0) return error.Unaligned;
    var args: gpu.Args = .{};
    args.add(x);
    args.add(@as(c_int, @backingInt(x_dtype)));
    args.add(@as(c_longlong, @intCast(ldx)));
    args.add(suh);
    args.add(xh);
    args.add(@as(c_int, @intCast(kdim)));
    try gpu.launch.launch(k.rot_in, .{ .grid = .{ .x = @intCast((kdim / 128 + 3) / 4), .y = @intCast(m) }, .block = .{ .x = 128 } }, stream, &args);
}

/// unpack: W_q [K, N] fp16 from the layer's trellis words, a warp a 16 x 16 tile.
pub fn unpack(k: *const Kernels, stream: gpu.Stream, l: weights.Linear, w: u64) !void {
    const f = k.unpack[l.k2][exl3_linear.codebook_mul1] orelse return error.MissingKernel;
    const s = exl3_linear.strides(l);
    var args: gpu.Args = .{};
    args.add(l.words);
    args.add(w);
    args.add(@as(c_int, @intCast(l.n)));
    args.add(@as(i64, @intCast(s[0])));
    args.add(@as(i64, @intCast(s[1])));
    try gpu.launch.launch(f, .{ .grid = .{ .x = l.n / 16, .y = l.k / 16 }, .block = .{ .x = 32 } }, stream, &args);
}

fn dtypeBytes(t: DType) usize {
    return switch (t) {
        .f32 => 4,
        else => 2,
    };
}

/// prefill.Workspace: one rotated input xh (fp16, at least M * K) and one W_q (fp16, at least K * N), each the
/// largest call's, reused by every call on the stream, and the bf16 Hadamard (hadamard()).
pub const Workspace = struct { xh: u64, w: u64, h: u64 };

/// prefill.matmul without a bias (no DeepSeek-V4.1 linear has one): out [M, N] (row stride os elements) =
/// x [M, K] (row stride ldx) @ W, the prompt path's arithmetic. With the kit's exl3.gemm for the layer's K2 (HIP), the
/// rotated rows go straight into it; else unpack writes W_q and the Triton GEMM reads it.
pub fn matmul(k: *const Kernels, t: tri.Tri, ws: Workspace, l: weights.Linear, x: u64, x_dtype: DType, ldx: usize, out: u64, out_type: tri_markov.OutType, os: usize, m: usize) !void {
    try rotIn(k, t.stream, x, x_dtype, ldx, l.suh, ws.xh, m, l.k);
    if (k.gemm[l.k2][gemmTile(l.n, m)] != null) return gemm(k, t.stream, l, ws.xh, out, out_type, os, m);
    try unpack(k, t.stream, l, ws.w);
    try tri_markov.matmul(t, ws.xh, ws.w, ws.h, l.svh, null, out, out_type, os, m, l.k, l.n);
}

/// exl3.gemm's row tile for N columns at m rows: 256 (index 1) while that still makes 120 programs, or from 512 rows
/// when 256-row tiles cover no more rows than 128-row ones do (each strip decoded half as often: a 1,024-row prompt
/// half's wq_a 0.59 -> 0.43 ms, wkv 0.29 -> 0.24, wo_a 0.41 -> 0.36; at 640 rows, 768 covered, it loses), else 128 (0).
/// Both tiles give the same bits.
pub fn gemmTile(n: usize, m: usize) usize {
    if ((n / 128) * ((m + 255) / 256) >= 120) return 1;
    return if (m >= 512 and (m + 255) / 256 * 2 == (m + 127) / 128) 1 else 0;
}

/// exl3.gemm: out [m, N] (row stride os, bf16 or fp32) = H128(xh @ W_q) / sqrt(128) * svh from the rotated rows xh
/// [m, K] (rot_in's) and the layer's words in the strips layout, a program a (row tile, 128-column strip).
pub fn gemm(k: *const Kernels, stream: gpu.Stream, l: weights.Linear, xh: u64, out: u64, out_type: tri_markov.OutType, os: usize, m: usize) !void {
    const tile = gemmTile(l.n, m);
    const f = k.gemm[l.k2][tile] orelse return error.MissingKernel;
    const s = exl3_linear.strides(l);
    const bm: usize = if (tile == 1) 256 else 128;
    var args: gpu.Args = .{};
    args.add(xh);
    args.add(l.words);
    args.add(@as(i64, s[0]));
    args.add(@as(i64, s[1]));
    args.add(l.svh);
    args.add(out);
    args.add(@as(i64, @intCast(os)));
    args.add(@as(c_int, @intCast(m)));
    args.add(@as(c_int, @intCast(l.k)));
    args.add(@as(c_int, @intCast(l.n)));
    args.add(@as(c_int, @intFromBool(out_type == .fp32)));
    try gpu.launch.launch(f, .{ .grid = .{ .x = @intCast((l.n / 128) * ((m + bm - 1) / bm)) }, .block = .{ .x = 256 } }, stream, &args);
}

/// prefill.Workspace.hadamard: H [128, 128] bf16, 1 - 2 * parity(i & j) (the bits of +-1.0).
pub fn hadamard(out: *[128 * 128]u16) void {
    for (0..128) |i| {
        for (0..128) |j| out[i * 128 + j] = if (@popCount(i & j) & 1 == 1) 0xbf80 else 0x3f80;
    }
}

test "the prompt GEMM's Hadamard: +-1, symmetric, orthogonal rows" {
    var h: [128 * 128]u16 = undefined;
    hadamard(&h);
    for (0..128) |j| try std.testing.expectEqual(@as(u16, 0x3f80), h[j]);
    for (0..128) |i| {
        for (0..128) |j| {
            try std.testing.expectEqual(h[i * 128 + j], h[j * 128 + i]);
            var dot: i32 = 0;
            for (0..128) |c| dot += @as(i32, if (h[i * 128 + c] == 0x3f80) 1 else -1) * @as(i32, if (h[j * 128 + c] == 0x3f80) 1 else -1);
            try std.testing.expectEqual(@as(i32, if (i == j) 128 else 0), dot);
        }
    }
}

test "unpack kernels by template arguments" {
    try std.testing.expectEqualDeep([2]u32{ 7, 2 }, template2("ILi7ELi2EEEvPKjP6__halfill").?);
    try std.testing.expectEqualDeep([2]u32{ 16, 2 }, template2("ILi16ELi2EEEvPKjP6__halfill").?);
    try std.testing.expectEqual(@as(?[2]u32, null), template2("v"));
}
