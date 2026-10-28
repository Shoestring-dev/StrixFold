//! Device memory, pinned and mapped host memory, each owned by one value; copies and fills, blocking or on a stream.
//! The surface and the usage counters are the CUDA runtime's (zig/src/cuda/memory.zig), so the family is unchanged;
//! device addresses are u64. Lifetime: a buffer is freed only after every stream that used it has finished with it
//! (tools/dsv41-hip/ADAPTER.md 3.4). hipFree synchronizes the device, so nothing is freed in steady state. The calls
//! without a stream are complete when they return (fills and device copies wait for the null stream).

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

/// Bytes this process holds in DeviceBuffers and HostBuffers, and the device peak since the last reset.
pub const Usage = struct { device: u64, host: u64, peak: u64 };

var device_bytes: std.atomic.Value(u64) = .init(0);
var host_bytes: std.atomic.Value(u64) = .init(0);
var peak_bytes: std.atomic.Value(u64) = .init(0);
var counts_mutex: std.atomic.Mutex = .unlocked;

/// The counts now; `reset_peak` starts a new peak at the current device bytes. Safe from any thread.
pub fn usage(reset_peak: bool) Usage {
    lockCounts();
    defer counts_mutex.unlock();
    const now = device_bytes.load(.monotonic);
    if (reset_peak) peak_bytes.store(now, .monotonic);
    return .{ .device = now, .host = host_bytes.load(.monotonic), .peak = @max(now, peak_bytes.load(.monotonic)) };
}

fn held(counter: *std.atomic.Value(u64), n: usize) void {
    lockCounts();
    defer counts_mutex.unlock();
    const now = counter.fetchAdd(n, .monotonic) + n;
    if (counter == &device_bytes) _ = peak_bytes.fetchMax(now, .monotonic);
}

fn freed(counter: *std.atomic.Value(u64), n: usize) void {
    lockCounts();
    defer counts_mutex.unlock();
    _ = counter.fetchSub(n, .monotonic);
}

fn lockCounts() void {
    while (!counts_mutex.tryLock()) std.Thread.yield() catch {};
}

/// Which allocator backs a DeviceBuffer (the G-c measurement picks; ADAPTER.md section 13 item 6).
pub const DeviceKind = enum(c_uint) {
    /// hipMalloc: coarse-grained; on gfx1151 GTT-backed system memory.
    default = abi.device_malloc_default,
    /// hipExtMallocWithFlags(hipDeviceMallocFinegrained): coherent with the host at system scope.
    finegrained = abi.device_malloc_finegrained,
    /// hipExtMallocWithFlags(hipDeviceMallocUncached).
    uncached = abi.device_malloc_uncached,
};

pub const DeviceBuffer = struct {
    d: *const Driver,
    ptr: abi.DevicePtr,
    len: usize,

    /// `len` bytes from hipMalloc (256-byte aligned or more); zero bytes allocate nothing and hold address 0.
    pub fn alloc(d: *const Driver, len: usize) Error!DeviceBuffer {
        return allocKind(d, len, .default);
    }

    pub fn allocKind(d: *const Driver, len: usize, kind: DeviceKind) Error!DeviceBuffer {
        var p: abi.Ptr = null;
        if (len > 0) {
            if (kind == .default) {
                try d.check(d.api.hipMalloc(&p, len), "hipMalloc");
            } else {
                try d.check(d.api.hipExtMallocWithFlags(&p, len, @intFromEnum(kind)), "hipExtMallocWithFlags");
            }
            if (p == null) return error.Invalid;
        }
        held(&device_bytes, len);
        return .{ .d = d, .ptr = abi.addr(p), .len = len };
    }

    /// Allocates and fills from host bytes in one call.
    pub fn fromHost(d: *const Driver, bytes: []const u8) Error!DeviceBuffer {
        var b = try alloc(d, bytes.len);
        errdefer b.free();
        try b.upload(0, bytes);
        return b;
    }

    pub fn free(self: *DeviceBuffer) void {
        if (self.ptr != 0) _ = self.d.api.hipFree(abi.ptr(self.ptr));
        freed(&device_bytes, self.len);
        self.* = undefined;
    }

    /// The device address `offset` bytes in; out of range is refused.
    pub fn at(self: DeviceBuffer, offset: usize) Error!abi.DevicePtr {
        if (offset > self.len) return error.Invalid;
        return self.ptr + offset;
    }

    fn span(self: DeviceBuffer, offset: usize, n: usize) Error!abi.DevicePtr {
        if (offset > self.len or n > self.len - offset) return error.Invalid;
        return self.ptr + offset;
    }

    pub fn upload(self: DeviceBuffer, offset: usize, bytes: []const u8) Error!void {
        const dst = try self.span(offset, bytes.len);
        if (bytes.len == 0) return;
        try copyH2D(self.d, dst, bytes);
    }

    pub fn download(self: DeviceBuffer, offset: usize, out: []u8) Error!void {
        const src = try self.span(offset, out.len);
        if (out.len == 0) return;
        try copyD2H(self.d, out, src);
    }

    /// Asynchronous only when `bytes` is pinned (HostBuffer), which must then stay unchanged until the stream passes
    /// this copy. Pageable bytes: use staging.zig's uploadStaged instead.
    pub fn uploadAsync(self: DeviceBuffer, offset: usize, bytes: []const u8, stream: abi.Stream) Error!void {
        const dst = try self.span(offset, bytes.len);
        if (bytes.len == 0) return;
        try self.d.check(self.d.api.hipMemcpyHtoDAsync(abi.ptr(dst), @constCast(bytes.ptr), bytes.len, stream), "hipMemcpyHtoDAsync");
    }

    pub fn downloadAsync(self: DeviceBuffer, offset: usize, out: []u8, stream: abi.Stream) Error!void {
        const src = try self.span(offset, out.len);
        if (out.len == 0) return;
        try self.d.check(self.d.api.hipMemcpyDtoHAsync(out.ptr, abi.ptr(src), out.len, stream), "hipMemcpyDtoHAsync");
    }

    pub fn copyFrom(self: DeviceBuffer, offset: usize, src: abi.DevicePtr, n: usize, stream: ?abi.Stream) Error!void {
        const dst = try self.span(offset, n);
        if (n == 0) return;
        if (stream) |s| {
            try copyD2DAsync(self.d, s, dst, src, n);
        } else {
            try self.d.check(self.d.api.hipMemcpyDtoD(abi.ptr(dst), abi.ptr(src), n), "hipMemcpyDtoD");
            try nullDone(self.d);
        }
    }

    pub fn fill8(self: DeviceBuffer, value: u8, stream: ?abi.Stream) Error!void {
        if (self.len == 0) return;
        if (stream) |s| {
            try fill8Async(self.d, s, self.ptr, value, self.len);
        } else {
            try self.d.check(self.d.api.hipMemsetD8(abi.ptr(self.ptr), value, self.len), "hipMemsetD8");
            try nullDone(self.d);
        }
    }

    /// Fills whole 32-bit words; the length must be a multiple of four.
    pub fn fill32(self: DeviceBuffer, value: u32, stream: ?abi.Stream) Error!void {
        if (self.len % 4 != 0) return error.Invalid;
        if (self.len == 0) return;
        if (stream) |s| {
            try fill32Async(self.d, s, self.ptr, value, self.len / 4);
        } else {
            try self.d.check(self.d.api.hipMemsetD32(abi.ptr(self.ptr), @bitCast(value), self.len / 4), "hipMemsetD32");
            try nullDone(self.d);
        }
    }
};

/// A blocking fill or device-to-device copy is complete when it returns: hipMemsetD8/D32 and hipMemcpyDtoD may return
/// with the work still queued on the null stream, which a non-blocking engine stream does not wait for (the smoke test
/// saw a kernel's output overwritten by an earlier "blocking" fill).
fn nullDone(d: *const Driver) Error!void {
    try d.check(d.api.hipStreamSynchronize(null), "hipStreamSynchronize(null)");
}

// -- copies and fills on raw device addresses (what replaces the family's direct driver-table calls) --------------

/// Blocking host-to-device copy (any host memory).
pub fn copyH2D(d: *const Driver, dst: abi.DevicePtr, bytes: []const u8) Error!void {
    if (bytes.len == 0) return;
    try d.check(d.api.hipMemcpyHtoD(abi.ptr(dst), @constCast(bytes.ptr), bytes.len), "hipMemcpyHtoD");
}

/// Blocking device-to-host copy (any host memory).
pub fn copyD2H(d: *const Driver, out: []u8, src: abi.DevicePtr) Error!void {
    if (out.len == 0) return;
    try d.check(d.api.hipMemcpyDtoH(out.ptr, abi.ptr(src), out.len), "hipMemcpyDtoH");
}

/// Device-to-device copy in `stream`'s order.
pub fn copyD2DAsync(d: *const Driver, stream: abi.Stream, dst: abi.DevicePtr, src: abi.DevicePtr, n: usize) Error!void {
    if (n == 0) return;
    try d.check(d.api.hipMemcpyDtoDAsync(abi.ptr(dst), abi.ptr(src), n, stream), "hipMemcpyDtoDAsync");
}

/// `n` bytes of `value` in `stream`'s order.
pub fn fill8Async(d: *const Driver, stream: abi.Stream, dst: abi.DevicePtr, value: u8, n: usize) Error!void {
    if (n == 0) return;
    try d.check(d.api.hipMemsetD8Async(abi.ptr(dst), value, n, stream), "hipMemsetD8Async");
}

/// `words` 32-bit words of `value` in `stream`'s order.
pub fn fill32Async(d: *const Driver, stream: abi.Stream, dst: abi.DevicePtr, value: u32, words: usize) Error!void {
    if (words == 0) return;
    try d.check(d.api.hipMemsetD32Async(abi.ptr(dst), @bitCast(value), words, stream), "hipMemsetD32Async");
}

/// hipHostMalloc flags, named (values checked against the headers by hip_abi_check.c).
pub const HostFlags = struct {
    portable: bool = true,
    mapped: bool = false,
    /// Fine-grained memory both sides see without cache maintenance; transport slabs ask for it explicitly.
    coherent: bool = false,
    write_combined: bool = false,

    fn bits(f: HostFlags) c_uint {
        var b: c_uint = 0;
        if (f.portable) b |= abi.host_malloc_portable;
        if (f.mapped) b |= abi.host_malloc_mapped;
        if (f.coherent) b |= abi.host_malloc_coherent else if (f.mapped) b |= abi.host_malloc_non_coherent;
        if (f.write_combined) b |= abi.host_malloc_write_combined;
        return b;
    }
};

/// Page-locked host memory: the only host memory an asynchronous copy can read or write without staging.
pub const HostBuffer = struct {
    d: *const Driver,
    bytes: []align(16) u8,
    /// Registered existing pages (unregister on free) rather than hipHostMalloc'd ones.
    registered: bool = false,

    pub fn alloc(d: *const Driver, len: usize) Error!HostBuffer {
        return allocFlags(d, len, .{});
    }

    /// Pinned memory kernels read and write directly (mapped into the device's address space). Non-coherent: use
    /// allocFlags(.{ .mapped = true, .coherent = true }) where both sides poll it.
    pub fn allocMapped(d: *const Driver, len: usize) Error!HostBuffer {
        return allocFlags(d, len, .{ .mapped = true });
    }

    pub fn allocFlags(d: *const Driver, len: usize, flags: HostFlags) Error!HostBuffer {
        if (len == 0) return error.Invalid;
        var p: abi.Ptr = null;
        try d.check(d.api.hipHostMalloc(&p, len, flags.bits()), "hipHostMalloc");
        if (p == null) return error.Invalid;
        held(&host_bytes, len);
        const base: [*]align(16) u8 = @ptrCast(@alignCast(p.?));
        return .{ .d = d, .bytes = base[0..len] };
    }

    /// Pins and maps existing pages (page-aligned, whole pages), e.g. a transport ring's mmap.
    pub fn register(d: *const Driver, bytes: []align(16) u8) Error!HostBuffer {
        if (bytes.len == 0) return error.Invalid;
        try d.check(d.api.hipHostRegister(bytes.ptr, bytes.len, abi.host_register_portable | abi.host_register_mapped), "hipHostRegister");
        held(&host_bytes, bytes.len);
        return .{ .d = d, .bytes = bytes, .registered = true };
    }

    /// The device address of a mapped buffer.
    pub fn device(self: HostBuffer) Error!abi.DevicePtr {
        var p: abi.Ptr = null;
        try self.d.check(self.d.api.hipHostGetDevicePointer(&p, self.bytes.ptr, 0), "hipHostGetDevicePointer");
        if (p == null) return error.Invalid;
        return abi.addr(p);
    }

    pub fn free(self: *HostBuffer) void {
        if (self.registered) {
            _ = self.d.api.hipHostUnregister(self.bytes.ptr);
        } else {
            _ = self.d.api.hipHostFree(self.bytes.ptr);
        }
        freed(&host_bytes, self.bytes.len);
        self.* = undefined;
    }

    pub fn slice(self: HostBuffer, comptime T: type) []T {
        return std.mem.bytesAsSlice(T, self.bytes[0 .. self.bytes.len / @sizeOf(T) * @sizeOf(T)]);
    }
};

test "usage counts held bytes and the device peak" {
    const before = usage(true);
    held(&device_bytes, 1000);
    held(&host_bytes, 24);
    freed(&device_bytes, 1000);
    const after = usage(false);
    try std.testing.expectEqual(before.device, after.device);
    try std.testing.expectEqual(before.host + 24, after.host);
    try std.testing.expectEqual(before.device + 1000, after.peak);
    try std.testing.expectEqual(before.device, usage(true).peak);
    freed(&host_bytes, 24);
}

test "host flags name exactly the bits asked for" {
    try std.testing.expectEqual(@as(c_uint, 0x1), (HostFlags{}).bits());
    try std.testing.expectEqual(@as(c_uint, 0x1 | 0x2 | 0x80000000), (HostFlags{ .mapped = true }).bits());
    try std.testing.expectEqual(@as(c_uint, 0x1 | 0x2 | 0x40000000), (HostFlags{ .mapped = true, .coherent = true }).bits());
}

test "out-of-range spans are refused before the runtime is called" {
    var d: Driver = undefined;
    const b: DeviceBuffer = .{ .d = &d, .ptr = 0x1000, .len = 64 };
    try std.testing.expectError(error.Invalid, b.at(65));
    try std.testing.expectEqual(@as(u64, 0x1040), try b.at(64));
    var out: [8]u8 = undefined;
    try std.testing.expectError(error.Invalid, b.download(60, &out));
    try std.testing.expectError(error.Invalid, b.upload(57, &out));
    try std.testing.expectError(error.Invalid, (DeviceBuffer{ .d = &d, .ptr = 0x1000, .len = 6 }).fill32(0, null));
}
