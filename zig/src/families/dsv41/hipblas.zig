//! hipBLAS by dlopen (libhipblas.so.3), for the fp32 GEMMs cublas.zig issues on CUDA: the same calls as torch issues
//! them (row-major through the transposed column-major call). For reproducible bytes the handle never uses atomics
//! (HIPBLAS_ATOMICS_NOT_ALLOWED), takes host scalars, and works in a fixed workspace from the engine's arena, which also
//! keeps its calls capturable into graphs. Values and prototypes are checked against the headers by hipblas_abi_check.c.
//! The vision MLP's bf16 linear (cublas.zig linearBf16) has no counterpart: the HIP build serves text only.
const std = @import("std");
const gpu = @import("gpu");

pub const Status = c_int;
pub const Handle = ?*opaque {};
/// hipblasOperation_t (hipBLAS's own numbering, not cuBLAS's 0/1).
pub const Op = enum(c_int) { n = 111, t = 112 };
const pointer_mode_host: c_int = 0;
const atomics_not_allowed: c_int = 0;
/// hipDataType's HIP_R_16BF and HIP_R_32F, hipblasComputeType_t's HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT (GemmEx).
const r_16bf: c_int = 14;
const r_32f: c_int = 0;
const compute_32f: c_int = 2;
const gemm_default: c_int = 160;

pub const Error = error{ LibraryUnavailable, MissingSymbol, BlasFailed };

const S = Status;
const P = ?*anyopaque;
const CP = ?*const anyopaque;

pub const Api = struct {
    hipblasCreate: *const fn (*Handle) callconv(.c) S,
    hipblasDestroy: *const fn (Handle) callconv(.c) S,
    hipblasSetStream: *const fn (Handle, gpu.abi.Stream) callconv(.c) S,
    hipblasSetWorkspace: *const fn (Handle, P, usize) callconv(.c) S,
    hipblasSetPointerMode: *const fn (Handle, c_int) callconv(.c) S,
    hipblasSetAtomicsMode: *const fn (Handle, c_int) callconv(.c) S,
    hipblasSgemm: *const fn (Handle, Op, Op, c_int, c_int, c_int, *const f32, CP, c_int, CP, c_int, *const f32, P, c_int) callconv(.c) S,
    hipblasGemmEx: *const fn (Handle, Op, Op, c_int, c_int, c_int, CP, CP, c_int, c_int, CP, c_int, c_int, CP, P, c_int, c_int, c_int, c_int) callconv(.c) S,
};

/// One handle bound to a stream, with torch's workspace size, no atomics, host scalars.
pub const Blas = struct {
    lib: std.DynLib,
    api: Api,
    handle: Handle = null,

    /// The workspace prompt.Chunk carves for it (cublas.zig's size: torch's default on the served GPU).
    pub const workspace_bytes: usize = 32 << 20;

    pub fn open(stream: gpu.Stream, workspace: u64) Error!Blas {
        var lib = std.DynLib.open("libhipblas.so.3") catch std.DynLib.open("libhipblas.so") catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse return error.MissingSymbol;
        }
        var b: Blas = .{ .lib = lib, .api = api };
        try check(api.hipblasCreate(&b.handle), "hipblasCreate");
        errdefer _ = api.hipblasDestroy(b.handle);
        try check(api.hipblasSetStream(b.handle, stream.handle), "hipblasSetStream");
        try check(api.hipblasSetWorkspace(b.handle, @ptrFromInt(workspace), workspace_bytes), "hipblasSetWorkspace");
        try check(api.hipblasSetPointerMode(b.handle, pointer_mode_host), "hipblasSetPointerMode");
        try check(api.hipblasSetAtomicsMode(b.handle, atomics_not_allowed), "hipblasSetAtomicsMode");
        return b;
    }

    pub fn close(b: *Blas) void {
        _ = b.api.hipblasDestroy(b.handle);
        b.lib.close();
        b.* = undefined;
    }

    fn check(s: Status, what: []const u8) Error!void {
        if (s != 0) {
            std.log.err("{s}: hipBLAS status {d}", .{ what, s });
            return error.BlasFailed;
        }
    }

    /// torch.matmul of x [rows, k] fp32 (row-major) and w [n, k] fp32 transposed: out [rows, n] fp32 (cublas.zig's
    /// call: column-major C^T = W x^T, transa T, transb N, m = n, n = rows, lda = ldb = k, ldc = n, alpha 1, beta 0).
    pub fn xwT(b: *const Blas, x: u64, w: u64, out: u64, rows: usize, k: usize, n: usize) Error!void {
        const one: f32 = 1;
        const zero: f32 = 0;
        try check(b.api.hipblasSgemm(b.handle, .t, .n, @intCast(n), @intCast(rows), @intCast(k), &one, @ptrFromInt(w), @intCast(k), @ptrFromInt(x), @intCast(k), &zero, @ptrFromInt(out), @intCast(n)), "hipblasSgemm");
    }

    /// out [rows, n] fp32 (rows ldo apart) = x [rows, k] bf16 (rows ldx apart) . w^T, or out + that when `accumulate`
    /// (beta 1), w bf16 [n, k] (dense.zig's prompt chunks: a dense projection's rank slice): xwT's column-major call
    /// through GemmEx, bf16 inputs, fp32 output and compute.
    pub fn gemmBf16(b: *const Blas, w: u64, x: u64, ldx: usize, out: u64, ldo: usize, rows: usize, k: usize, n: usize, accumulate: bool) Error!void {
        const one: f32 = 1;
        const beta: f32 = if (accumulate) 1 else 0;
        try check(b.api.hipblasGemmEx(b.handle, .t, .n, @intCast(n), @intCast(rows), @intCast(k), &one, @ptrFromInt(w), r_16bf, @intCast(k), @ptrFromInt(x), r_16bf, @intCast(ldx), &beta, @ptrFromInt(out), r_32f, @intCast(ldo), compute_32f, gemm_default), "hipblasGemmEx");
    }

    /// torch.matmul of x [rows, k] fp32 and v [1, k] fp32 transposed (the DSpark confidence head): out [rows] fp32
    /// (cublas.zig's call: C^T = v^T x^T untransposed, m = 1, n = rows, lda = 1, ldb = k, ldc = 1).
    pub fn xv(b: *const Blas, x: u64, v: u64, out: u64, rows: usize, k: usize) Error!void {
        const one: f32 = 1;
        const zero: f32 = 0;
        try check(b.api.hipblasSgemm(b.handle, .n, .n, 1, @intCast(rows), @intCast(k), &one, @ptrFromInt(v), 1, @ptrFromInt(x), @intCast(k), &zero, @ptrFromInt(out), 1), "hipblasSgemm");
    }
};

extern fn tf_hipblas_abi_checked() c_int;

test "hipBLAS prototypes and values agree with the installed headers" {
    try std.testing.expectEqual(@as(c_int, 8), tf_hipblas_abi_checked());
}
