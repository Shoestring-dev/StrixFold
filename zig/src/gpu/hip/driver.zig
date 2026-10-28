//! The HIP runtime opened at run time: libamdhip64.so.7's entry points in one table, checked calls, and the process's
//! device-fault state. Named Driver, as the CUDA runtime's handle is, so the family's `d: *const Driver` is unchanged.

const std = @import("std");
const abi = @import("abi.zig");

pub const Error = error{
    DriverUnavailable,
    MissingSymbol,
    GpuFailed,
    OutOfDeviceMemory,
    NotReady,
    NotFound,
    Invalid,
    Unsupported,
    DeviceFault,
    Timeout,
    InCapture,
    NoKernelImage,
};

/// The first device-fatal result this process saw (0: none). A faulted device fails every later call, so a rank reads
/// this to tell its peer why it stops instead of hanging on a collective (TensorFold-recipe #25).
var fault_code: std.atomic.Value(i32) = .init(0);

pub fn health() ?abi.Result {
    const c = fault_code.load(.acquire);
    return if (c == 0) null else c;
}

pub fn markFault(res: abi.Result) void {
    _ = fault_code.cmpxchgStrong(0, res, .acq_rel, .acquire);
}

/// What a non-success result means to a caller; null for success. Pure, so tests need no runtime.
pub fn classify(res: abi.Result) ?Error {
    return switch (res) {
        abi.success => null,
        abi.error_out_of_memory => error.OutOfDeviceMemory,
        abi.error_not_ready => error.NotReady,
        abi.error_not_found => error.NotFound,
        abi.error_invalid_device_function, abi.error_invalid_image, abi.error_no_binary_for_gpu => error.NoKernelImage,
        abi.error_stream_capture_unsupported => error.InCapture,
        // the sticky results a kernel fault leaves: every later call on the device fails
        abi.error_ecc_not_correctable, abi.error_illegal_address, abi.error_launch_timeout, abi.error_assert, abi.error_launch_failure => error.DeviceFault,
        else => error.GpuFailed,
    };
}

/// Every field of `abi.Api` from `lib`, or the first name it lacks.
fn resolve(lib: *std.DynLib) error{MissingSymbol}!abi.Api {
    var api: abi.Api = undefined;
    const info = @typeInfo(abi.Api).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(api, name) = lib.lookup(T, name) orelse {
            last_missing = name;
            return error.MissingSymbol;
        };
    }
    return api;
}

var last_missing: []const u8 = "";

/// hipGetDevicePropertiesR0600, called only through device_arch.c with the header's hipDeviceProp_t.
pub const Properties = *const fn (*anyopaque, c_int) callconv(.c) abi.Result;

pub const Driver = struct {
    lib: std.DynLib,
    api: abi.Api,
    properties: Properties,

    pub fn open() Error!Driver {
        return openPath("libamdhip64.so.7");
    }

    /// Resolves every field of `abi.Api` by its exact name (a missing symbol refuses the whole runtime), then hipInit.
    pub fn openPath(path: []const u8) Error!Driver {
        var lib = std.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        const api = resolve(&lib) catch {
            std.log.err("{s} has no {s}", .{ path, last_missing });
            return error.MissingSymbol;
        };
        const properties = lib.lookup(Properties, "hipGetDevicePropertiesR0600") orelse {
            std.log.err("{s} has no hipGetDevicePropertiesR0600", .{path});
            return error.MissingSymbol;
        };
        const d: Driver = .{ .lib = lib, .api = api, .properties = properties };
        try d.check(api.hipInit(0), "hipInit");
        return d;
    }

    pub fn close(self: *Driver) void {
        self.lib.close();
    }

    /// Logs a failed call with HIP's own name and text for the result; NOT_READY is a state, not a failure. A
    /// device-fatal result is also recorded in `health()`.
    pub fn check(self: *const Driver, res: abi.Result, what: []const u8) Error!void {
        const e = classify(res) orelse return;
        if (e == error.NotReady) return e;
        if (e == error.DeviceFault) markFault(res);
        std.log.err("{s}: {s} ({d}) {s}", .{ what, self.errorName(res), res, self.errorText(res) });
        return e;
    }

    pub fn errorName(self: *const Driver, res: abi.Result) []const u8 {
        return if (self.api.hipGetErrorName(res)) |p| std.mem.span(p) else "hipErrorUnknown";
    }

    pub fn errorText(self: *const Driver, res: abi.Result) []const u8 {
        return if (self.api.hipGetErrorString(res)) |p| std.mem.span(p) else "";
    }

    /// HIP's runtime version (HIP_VERSION: major * 10^7 + minor * 10^5 + patch).
    pub fn version(self: *const Driver) Error!c_int {
        var v: c_int = 0;
        try self.check(self.api.hipRuntimeGetVersion(&v), "hipRuntimeGetVersion");
        return v;
    }

    pub fn deviceCount(self: *const Driver) Error!c_int {
        var n: c_int = 0;
        try self.check(self.api.hipGetDeviceCount(&n), "hipGetDeviceCount");
        return n;
    }
};

test "results map to the errors a caller acts on" {
    try std.testing.expectEqual(@as(?Error, null), classify(0));
    try std.testing.expectEqual(@as(?Error, error.OutOfDeviceMemory), classify(2));
    try std.testing.expectEqual(@as(?Error, error.NotReady), classify(600));
    try std.testing.expectEqual(@as(?Error, error.NoKernelImage), classify(209));
    try std.testing.expectEqual(@as(?Error, error.NoKernelImage), classify(98));
    try std.testing.expectEqual(@as(?Error, error.DeviceFault), classify(700));
    try std.testing.expectEqual(@as(?Error, error.DeviceFault), classify(719));
    try std.testing.expectEqual(@as(?Error, error.DeviceFault), classify(710));
    try std.testing.expectEqual(@as(?Error, error.GpuFailed), classify(701)); // out of resources: not sticky
    try std.testing.expectEqual(@as(?Error, error.InCapture), classify(900));
    try std.testing.expectEqual(@as(?Error, error.GpuFailed), classify(1));
}

test "the first device fault is kept" {
    try std.testing.expectEqual(@as(?abi.Result, null), health());
    markFault(700);
    markFault(719);
    try std.testing.expectEqual(@as(?abi.Result, 700), health());
    fault_code.store(0, .release);
}

test "a library without the runtime's symbols is refused whole" {
    try std.testing.expectError(error.DriverUnavailable, Driver.openPath("/nonexistent/libamdhip64.so.7"));
    var libc = try std.DynLib.open("libc.so.6");
    defer libc.close();
    try std.testing.expectError(error.MissingSymbol, resolve(&libc));
    try std.testing.expectEqualStrings("hipInit", last_missing);
}
