//! Dense bf16 projections (weights.Linear.dense: a fine-tuned pack's attn.wo_b and shared expert's w2 in
//! layers 10-36, bf16 [N, K] by the rank's input rows): a decode / verify window's rows through dense.hip's GEMV (each
//! row's arithmetic independent of the window: the EXL3 decode paths' row invariance), a prompt chunk's through
//! hipBLAS GemmEx (bf16 in, fp32 compute and out), and the shared expert's activation (dense.hip's swiglu, the
//! reference Expert.forward's clamps and silu in fp32, then bf16).
const std = @import("std");
const gpu = @import("gpu");
const weights = @import("weights.zig");
const blas = @import("blas.zig");

/// dense.hip's kernels (hip_load.zig binds them; absent from a kit without the dense group).
pub const Kernels = struct { gemv: gpu.Function, swiglu: gpu.Function, edit: ?gpu.Function = null }; // edit: newer kits

/// The output columns a GEMV block owns (dense.hip: 8 waves of 4).
pub const block_cols = 32;

/// y [rows, l.n] fp32 (rows ldy floats apart) = x [rows, l.k] bf16 (rows ldx elements apart) . l^T, or y + that when
/// `accumulate` (one fp32 add a value).
pub fn gemv(k: *const Kernels, s: gpu.Stream, l: weights.Linear, x: u64, ldx: usize, y: u64, ldy: usize, rows: usize, accumulate: bool) !void {
    if (l.dense == 0) return error.NotDense;
    if (rows == 0) return;
    if (l.k % 128 != 0 or l.dense % 16 != 0 or ldx < l.k or ldy < l.n) return error.UnsupportedShape;
    var a: gpu.Args = .{};
    a.add(l.dense);
    a.add(x);
    a.add(y);
    a.add(@as(i32, @intCast(l.k)));
    a.add(@as(i32, @intCast(l.n)));
    a.add(@as(i32, @intCast(rows)));
    a.add(@as(i64, @intCast(ldx)));
    a.add(@as(i64, @intCast(ldy)));
    a.add(@as(i32, @intFromBool(accumulate)));
    try gpu.launch.launch(k.gemv, .{ .grid = .{ .x = @intCast((l.n + block_cols - 1) / block_cols) }, .block = .{ .x = 256 } }, s, &a);
}

/// The same product for a prompt chunk: hipBLAS GemmEx (HIP builds only).
pub fn gemm(b: *const blas.Blas, l: weights.Linear, x: u64, ldx: usize, y: u64, ldy: usize, rows: usize, accumulate: bool) !void {
    if (l.dense == 0) return error.NotDense;
    if (rows == 0) return;
    if (comptime gpu.backend == .hip) {
        try b.gemmBf16(l.dense, x, ldx, y, ldy, rows, l.k, l.n, accumulate);
    } else return error.NotPortedYet;
}

/// h [rows, width] bf16 (rows ldh apart) = silu(min(g, limit)) * clamp(u, -limit, limit) of g, u fp32 (rows ld apart).
pub fn swiglu(k: *const Kernels, s: gpu.Stream, g: u64, u: u64, ld: usize, h: u64, ldh: usize, rows: usize, width: usize, limit: f32) !void {
    if (rows == 0) return;
    var a: gpu.Args = .{};
    a.add(g);
    a.add(u);
    a.add(@as(i64, @intCast(ld)));
    a.add(h);
    a.add(@as(i64, @intCast(ldh)));
    a.add(@as(i32, @intCast(rows)));
    a.add(@as(i32, @intCast(width)));
    a.add(limit);
    const total = rows * width;
    try gpu.launch.launch(k.swiglu, .{ .grid = .{ .x = @intCast(@min((total + 255) / 256, 4096)) }, .block = .{ .x = 256 } }, s, &a);
}

/// out (+)= g * (y - alpha r (r . y)) for each fp32 row of y (edit.zig); out may be y when not accumulating.
pub fn edit(k: *const Kernels, s: gpu.Stream, ed: weights.Edit, y: u64, ldy: usize, out: u64, ldo: usize, n: usize, rows: usize, accumulate: bool) !void {
    const f = k.edit orelse return error.NoEditKernel;
    if (rows == 0) return;
    if (ldy < n or ldo < n) return error.UnsupportedShape;
    var a: gpu.Args = .{};
    a.add(ed.r);
    a.add(ed.g);
    a.add(ed.alpha);
    a.add(y);
    a.add(@as(i64, @intCast(ldy)));
    a.add(out);
    a.add(@as(i64, @intCast(ldo)));
    a.add(@as(i32, @intCast(n)));
    a.add(@as(i32, @intFromBool(accumulate)));
    try gpu.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
}
