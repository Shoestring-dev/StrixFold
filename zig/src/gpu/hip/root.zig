//! TensorFold's HIP runtime for the dsv41 family: libamdhip64 through dlopen, with the CUDA runtime's surface
//! (zig/src/cuda/root.zig) so a family written against it runs unchanged. Leaf modules after ashhart/TensorFold
//! d46d17a zig/src/hip (Apache-2.0); design in tools/dsv41-hip/ADAPTER.md.

pub const abi = @import("abi.zig");
pub const Driver = @import("driver.zig").Driver;
pub const Error = @import("driver.zig").Error;
pub const health = @import("driver.zig").health;
pub const markFault = @import("driver.zig").markFault;
pub const Context = @import("context.zig").Context;
pub const caps = @import("caps.zig");
pub const Caps = caps.Caps;
pub const memory = @import("memory.zig");
pub const DeviceBuffer = memory.DeviceBuffer;
pub const DeviceKind = memory.DeviceKind;
pub const HostBuffer = memory.HostBuffer;
pub const HostFlags = memory.HostFlags;
pub const usage = memory.usage;
pub const Usage = memory.Usage;
pub const copyH2D = memory.copyH2D;
pub const copyD2H = memory.copyD2H;
pub const copyD2DAsync = memory.copyD2DAsync;
pub const fill8Async = memory.fill8Async;
pub const fill32Async = memory.fill32Async;
pub const Stream = @import("stream.zig").Stream;
pub const Event = @import("stream.zig").Event;
pub const Module = @import("module.zig").Module;
pub const Function = @import("module.zig").Function;
pub const checkImage = @import("module.zig").checkImage;
pub const code_object = @import("code_object.zig");
pub const launch = @import("launch.zig");
pub const Args = launch.Args;
pub const Config = launch.Config;
pub const Dim3 = launch.Dim3;
pub const graph = @import("graph.zig");
pub const kit = @import("kit.zig");
pub const aot = @import("aot.zig");

/// Linked from hip_abi_check.c: its static assertions held when this binary was built.
extern fn tf_hip_abi_checked() c_int;

pub fn abiChecked() c_int {
    return tf_hip_abi_checked();
}

test {
    _ = abi;
    _ = @import("driver.zig");
    _ = caps;
    _ = memory;
    _ = @import("stream.zig");
    _ = @import("module.zig");
    _ = code_object;
    _ = launch;
    _ = graph;
    _ = kit;
    _ = aot;
    refAll(@This());
}

/// Every declaration, and every declaration of every struct among them, so methods no test calls still compile.
fn refAll(comptime T: type) void {
    const std = @import("std");
    inline for (comptime std.meta.declarations(T)) |name| {
        const v = &@field(T, name);
        if (@TypeOf(v.*) == type) switch (@typeInfo(v.*)) {
            .@"struct", .@"union", .@"enum" => if (v.* != T) refAll(v.*),
            else => {},
        };
    }
}
