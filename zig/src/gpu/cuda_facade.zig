//! The `gpu` module of a CUDA build: the untouched CUDA runtime (zig/src/cuda) under the backend-neutral names the dsv41
//! family uses (interface.zig). The added functions issue exactly the driver calls the family made directly before
//! (cuMemcpy*_v2, cuMemsetD*Async), so a CUDA build behaves as it did. Kept compiling host-only until first light
//! (ADAPTER.md section 13 item 1); not qualified on any GPU in this port.

const std = @import("std");
const cuda = @import("cuda");

pub const Backend = enum { cuda, hip };
pub const backend: Backend = .cuda;

pub const features = .{ .pdl = true, .clusters = true, .vision = true, .verbs_rdma = true, .l2_prefetch = true, .world4 = true };

pub const abi = cuda.abi;
pub const Driver = cuda.Driver;
pub const Error = cuda.Error || error{ Unsupported, InCapture, Timeout, DeviceFault, NoKernelImage };
pub const Context = cuda.Context;
pub const DeviceBuffer = cuda.DeviceBuffer;
pub const HostBuffer = cuda.HostBuffer;
pub const Usage = cuda.Usage;
pub const usage = cuda.usage;
pub const Stream = cuda.Stream;
pub const Event = cuda.Event;
pub const Module = cuda.Module;
pub const Function = cuda.Function;
pub const launch = cuda.launch;
pub const Args = cuda.Args;
pub const Config = cuda.Config;
pub const Dim3 = cuda.Dim3;
pub const graph = cuda.graph;
/// CUDA-only pieces the family's CUDA paths keep using as before.
pub const aot = cuda.aot;
pub const kernels = cuda.kernels;
pub const nccl = cuda.nccl;
pub const cublaslt = cuda.cublaslt;

pub fn copyH2D(d: *const Driver, dst: u64, bytes: []const u8) Error!void {
    if (bytes.len == 0) return;
    try d.check(d.api.cuMemcpyHtoD_v2(dst, bytes.ptr, bytes.len), "cuMemcpyHtoD");
}

pub fn copyD2H(d: *const Driver, out: []u8, src: u64) Error!void {
    if (out.len == 0) return;
    try d.check(d.api.cuMemcpyDtoH_v2(out.ptr, src, out.len), "cuMemcpyDtoH");
}

pub fn copyD2DAsync(d: *const Driver, stream: abi.Stream, dst: u64, src: u64, n: usize) Error!void {
    if (n == 0) return;
    try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, n, stream), "cuMemcpyDtoDAsync");
}

pub fn fill8Async(d: *const Driver, stream: abi.Stream, dst: u64, value: u8, n: usize) Error!void {
    if (n == 0) return;
    try d.check(d.api.cuMemsetD8Async(dst, value, n, stream), "cuMemsetD8Async");
}

pub fn fill32Async(d: *const Driver, stream: abi.Stream, dst: u64, value: u32, words: usize) Error!void {
    if (words == 0) return;
    try d.check(d.api.cuMemsetD32Async(dst, value, words, stream), "cuMemsetD32Async");
}

pub fn uploadPinned(d: *const Driver, stream: Stream, dst: u64, pinned: []const u8) Error!void {
    if (pinned.len == 0) return;
    try d.check(d.api.cuMemcpyHtoDAsync_v2(dst, pinned.ptr, pinned.len, stream.handle), "cuMemcpyHtoDAsync");
}

pub fn streamCapturing(s: Stream) Error!bool {
    return (try graph.captureStatus(s)) != .none;
}

pub fn synchronizeFor(s: Stream, ns: u64) Error!void {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    const start = @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    while (!try s.done()) {
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        const now = @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        if (now - start >= ns) return error.Timeout;
        std.Thread.yield() catch {};
    }
}

/// CUDA records no device fault state of its own here.
pub fn health() ?abi.Result {
    return null;
}

/// CUDA stages pageable host memory itself before cuMemcpyHtoDAsync returns, so the CUDA build keeps its direct call.
pub const Staging = struct {
    d: *const Driver,

    pub fn init(d: *const Driver, _: usize) Error!Staging {
        return .{ .d = d };
    }

    pub fn deinit(s: *Staging) void {
        s.* = undefined;
    }

    pub fn upload(s: *Staging, stream: Stream, dst: u64, bytes: []const u8) Error!void {
        try uploadPinned(s.d, stream, dst, bytes);
    }
};

test {
    @import("interface.zig").check(@This());
}
