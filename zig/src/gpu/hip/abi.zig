//! HIP 7 runtime entry points declared by hand (after ashhart/TensorFold d46d17a zig/src/hip/abi.zig); every value and
//! prototype here is checked against the installed headers by hip_abi_check.c, so a ROCm upgrade that changes one fails
//! the build. HIP's own pointers stay `Ptr` here; the backend hands the family u64 device addresses.

const std = @import("std");

pub const Result = c_int;
pub const Ptr = ?*anyopaque;
/// A device address as the family passes it (the CUDA backend's CUdeviceptr is a u64 too).
pub const DevicePtr = u64;
pub const Stream = ?*opaque {};
pub const Event = ?*opaque {};
pub const Module = ?*opaque {};
pub const Function = ?*opaque {};
pub const Graph = ?*opaque {};
pub const GraphNode = ?*opaque {};
pub const GraphExec = ?*opaque {};

pub const success: Result = 0;
pub const error_out_of_memory: Result = 2;
pub const error_invalid_image: Result = 200;
pub const error_no_binary_for_gpu: Result = 209;
pub const error_invalid_device_function: Result = 98;
pub const error_not_found: Result = 500;
pub const error_not_ready: Result = 600;
pub const error_ecc_not_correctable: Result = 214;
pub const error_illegal_address: Result = 700;
pub const error_launch_timeout: Result = 702;
pub const error_assert: Result = 710;
pub const error_launch_failure: Result = 719;
pub const error_stream_capture_unsupported: Result = 900;

pub const stream_non_blocking: c_uint = 1;
pub const event_disable_timing: c_uint = 2;

pub const host_malloc_portable: c_uint = 0x1;
pub const host_malloc_mapped: c_uint = 0x2;
pub const host_malloc_write_combined: c_uint = 0x4;
pub const host_malloc_coherent: c_uint = 0x40000000;
pub const host_malloc_non_coherent: c_uint = 0x80000000;
pub const host_register_portable: c_uint = 0x1;
pub const host_register_mapped: c_uint = 0x2;
/// hipExtMallocWithFlags: hipDeviceMallocDefault, hipDeviceMallocFinegrained, hipDeviceMallocUncached.
pub const device_malloc_default: c_uint = 0x0;
pub const device_malloc_finegrained: c_uint = 0x1;
pub const device_malloc_uncached: c_uint = 0x3;

pub const CaptureMode = enum(c_int) { global = 0, thread_local = 1, relaxed = 2 };
pub const CaptureStatus = enum(c_int) { none = 0, active = 1, invalidated = 2, _ };

/// hipDeviceAttribute_t: HIP's numbering, not CUDA's.
pub const DeviceAttribute = enum(c_int) {
    clock_rate = 5,
    cooperative_launch = 10,
    integrated = 16,
    l2_cache_size = 19,
    max_threads_per_block = 56,
    multiprocessor_count = 63,
    max_shared_memory_per_block = 74,
    max_shared_memory_per_block_optin = 75,
    warp_size = 87,
    max_shared_memory_per_multiprocessor = 10002,
    wall_clock_rate = 10017,
};

/// hipFunction_attribute.
pub const FunctionAttribute = enum(c_int) {
    max_threads_per_block = 0,
    shared_size_bytes = 1,
    const_size_bytes = 2,
    local_size_bytes = 3,
    num_regs = 4,
    max_dynamic_shared_size_bytes = 8,
};

pub const Dim3 = extern struct { x: c_uint = 1, y: c_uint = 1, z: c_uint = 1 };

/// hipKernelNodeParams: HIP's own field order (block first), 64 bytes.
pub const KernelNodeParams = extern struct {
    block: Dim3,
    extra: ?[*]?*anyopaque,
    func: Function,
    grid: Dim3,
    params: ?[*]?*anyopaque,
    shared_bytes: c_uint,
};

pub const ExecUpdateResult = enum(c_int) {
    success = 0,
    @"error" = 1,
    topology_changed = 2,
    node_type_changed = 3,
    function_changed = 4,
    parameters_changed = 5,
    not_supported = 6,
    unsupported_function_change = 7,
    _,
};

const R = Result;
const P = Ptr;
const CP = ?*const anyopaque;
pub const Params = ?[*]?*anyopaque; // a launch's kernel arguments (or extra), as hipModuleLaunchKernel takes them

/// Each field is the exact exported symbol of libamdhip64.so.7.
pub const Api = struct {
    hipInit: *const fn (c_uint) callconv(.c) R,
    hipRuntimeGetVersion: *const fn (*c_int) callconv(.c) R,
    hipGetDeviceCount: *const fn (*c_int) callconv(.c) R,
    hipSetDevice: *const fn (c_int) callconv(.c) R,
    hipGetDevice: *const fn (*c_int) callconv(.c) R,
    hipDeviceGetName: *const fn ([*]u8, c_int, c_int) callconv(.c) R,
    hipDeviceGetAttribute: *const fn (*c_int, DeviceAttribute, c_int) callconv(.c) R,
    hipDeviceSynchronize: *const fn () callconv(.c) R,
    hipMemGetInfo: *const fn (*usize, *usize) callconv(.c) R,
    hipGetErrorName: *const fn (R) callconv(.c) ?[*:0]const u8,
    hipGetErrorString: *const fn (R) callconv(.c) ?[*:0]const u8,
    hipMalloc: *const fn (*P, usize) callconv(.c) R,
    hipExtMallocWithFlags: *const fn (*P, usize, c_uint) callconv(.c) R,
    hipFree: *const fn (P) callconv(.c) R,
    hipHostMalloc: *const fn (*P, usize, c_uint) callconv(.c) R,
    hipHostFree: *const fn (P) callconv(.c) R,
    hipHostRegister: *const fn (P, usize, c_uint) callconv(.c) R,
    hipHostUnregister: *const fn (P) callconv(.c) R,
    hipHostGetDevicePointer: *const fn (*P, P, c_uint) callconv(.c) R,
    hipMemcpyHtoD: *const fn (P, P, usize) callconv(.c) R,
    hipMemcpyDtoH: *const fn (P, P, usize) callconv(.c) R,
    hipMemcpyDtoD: *const fn (P, P, usize) callconv(.c) R,
    hipMemcpyHtoDAsync: *const fn (P, P, usize, Stream) callconv(.c) R,
    hipMemcpyDtoHAsync: *const fn (P, P, usize, Stream) callconv(.c) R,
    hipMemcpyDtoDAsync: *const fn (P, P, usize, Stream) callconv(.c) R,
    hipMemsetD8: *const fn (P, u8, usize) callconv(.c) R,
    hipMemsetD8Async: *const fn (P, u8, usize, Stream) callconv(.c) R,
    hipMemsetD32: *const fn (P, c_int, usize) callconv(.c) R,
    hipMemsetD32Async: *const fn (P, c_int, usize, Stream) callconv(.c) R,
    hipStreamCreateWithFlags: *const fn (*Stream, c_uint) callconv(.c) R,
    hipStreamDestroy: *const fn (Stream) callconv(.c) R,
    hipStreamSynchronize: *const fn (Stream) callconv(.c) R,
    hipStreamQuery: *const fn (Stream) callconv(.c) R,
    hipStreamWaitEvent: *const fn (Stream, Event, c_uint) callconv(.c) R,
    hipStreamIsCapturing: *const fn (Stream, *CaptureStatus) callconv(.c) R,
    hipEventCreateWithFlags: *const fn (*Event, c_uint) callconv(.c) R,
    hipEventDestroy: *const fn (Event) callconv(.c) R,
    hipEventRecord: *const fn (Event, Stream) callconv(.c) R,
    hipEventSynchronize: *const fn (Event) callconv(.c) R,
    hipEventQuery: *const fn (Event) callconv(.c) R,
    hipEventElapsedTime: *const fn (*f32, Event, Event) callconv(.c) R,
    hipModuleLoadData: *const fn (*Module, CP) callconv(.c) R,
    hipModuleUnload: *const fn (Module) callconv(.c) R,
    hipModuleGetFunction: *const fn (*Function, Module, [*:0]const u8) callconv(.c) R,
    hipModuleGetGlobal: *const fn (*P, *usize, Module, [*:0]const u8) callconv(.c) R,
    hipFuncGetAttribute: *const fn (*c_int, FunctionAttribute, Function) callconv(.c) R,
    hipModuleOccupancyMaxActiveBlocksPerMultiprocessor: *const fn (*c_int, Function, c_int, usize) callconv(.c) R,
    hipModuleLaunchKernel: *const fn (Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Stream, Params, Params) callconv(.c) R,
    hipStreamBeginCapture: *const fn (Stream, CaptureMode) callconv(.c) R,
    hipStreamEndCapture: *const fn (Stream, *Graph) callconv(.c) R,
    hipGraphCreate: *const fn (*Graph, c_uint) callconv(.c) R,
    hipGraphDestroy: *const fn (Graph) callconv(.c) R,
    hipGraphAddKernelNode: *const fn (*GraphNode, Graph, ?[*]const GraphNode, usize, *const KernelNodeParams) callconv(.c) R,
    hipGraphKernelNodeSetParams: *const fn (GraphNode, *const KernelNodeParams) callconv(.c) R,
    hipGraphAddDependencies: *const fn (Graph, [*]const GraphNode, [*]const GraphNode, usize) callconv(.c) R,
    hipGraphGetNodes: *const fn (Graph, ?[*]GraphNode, *usize) callconv(.c) R,
    hipGraphInstantiateWithFlags: *const fn (*GraphExec, Graph, c_ulonglong) callconv(.c) R,
    hipGraphUpload: *const fn (GraphExec, Stream) callconv(.c) R,
    hipGraphLaunch: *const fn (GraphExec, Stream) callconv(.c) R,
    hipGraphExecDestroy: *const fn (GraphExec) callconv(.c) R,
    hipGraphExecUpdate: *const fn (GraphExec, Graph, *GraphNode, *ExecUpdateResult) callconv(.c) R,
    hipGraphExecKernelNodeSetParams: *const fn (GraphExec, GraphNode, *const KernelNodeParams) callconv(.c) R,
};

pub fn ptr(a: DevicePtr) Ptr {
    return @ptrFromInt(a);
}

pub fn addr(p: Ptr) DevicePtr {
    return @intFromPtr(p);
}

test "HIP handles and kernel node parameters keep C widths and HIP's layout" {
    try std.testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(Stream));
    try std.testing.expectEqual(@sizeOf(c_int), @sizeOf(DeviceAttribute));
    try std.testing.expectEqual(64, @sizeOf(KernelNodeParams));
    try std.testing.expectEqual(0, @offsetOf(KernelNodeParams, "block"));
    try std.testing.expectEqual(16, @offsetOf(KernelNodeParams, "extra"));
    try std.testing.expectEqual(24, @offsetOf(KernelNodeParams, "func"));
    try std.testing.expectEqual(32, @offsetOf(KernelNodeParams, "grid"));
    try std.testing.expectEqual(48, @offsetOf(KernelNodeParams, "params"));
    try std.testing.expectEqual(56, @offsetOf(KernelNodeParams, "shared_bytes"));
}
