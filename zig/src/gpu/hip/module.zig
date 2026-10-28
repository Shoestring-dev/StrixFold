//! Loaded GPU code: raw gfx code objects built for this exact device (never a CUBIN, fatbin or another gfx's image),
//! their kernels by their stable extern "C" names and their device globals. Kernels are normally reached through the
//! manifest kit (kit.zig), which checks each object's hash, arch and argument metadata before this load.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

const hsaco = @import("../hsaco.zig");
pub const em_amdgpu = hsaco.em_amdgpu;
pub const ef_amdgpu_mach_mask = 0xff;
pub const machOf = hsaco.machOf;

/// The image is a 64-bit little-endian AMDGPU ELF for exactly `arch` (an offload bundle or another gfx is refused).
pub fn checkImage(image: []const u8, arch: []const u8) Error!void {
    if (image.len < 64 or !std.mem.eql(u8, image[0..4], "\x7fELF") or image[4] != 2 or image[5] != 1) return error.NoKernelImage;
    if (std.mem.readInt(u16, image[18..20], .little) != em_amdgpu) return error.NoKernelImage;
    const want = machOf(arch) orelse return error.Unsupported;
    if (std.mem.readInt(u32, image[48..52], .little) & ef_amdgpu_mach_mask != want) return error.NoKernelImage;
}

pub const Module = struct {
    d: *const Driver,
    handle: abi.Module,

    /// A code object (8-byte aligned); the runtime copies what it needs, so the bytes may go after this returns.
    pub fn load(d: *const Driver, image: []const u8) Error!Module {
        if (image.len == 0) {
            std.log.err("empty GPU image: this binary or kit has no code object for it", .{});
            return error.Invalid;
        }
        if (@intFromPtr(image.ptr) % 8 != 0) return error.Invalid;
        var m: abi.Module = null;
        try d.check(d.api.hipModuleLoadData(&m, image.ptr), "hipModuleLoadData");
        if (m == null) return error.Invalid;
        return .{ .d = d, .handle = m };
    }

    pub fn unload(self: *Module) void {
        _ = self.d.api.hipModuleUnload(self.handle);
        self.* = undefined;
    }

    /// A kernel by its exact extern "C" name; the function lives as long as the module.
    pub fn function(self: Module, name: [:0]const u8) Error!Function {
        var f: abi.Function = null;
        self.d.check(self.d.api.hipModuleGetFunction(&f, self.handle, name.ptr), "hipModuleGetFunction") catch |e| {
            std.log.err("kernel symbol not in module: {s}", .{name});
            return e;
        };
        if (f == null) return error.Invalid;
        return .{ .d = self.d, .handle = f };
    }

    pub const Global = struct { ptr: abi.DevicePtr, len: usize };

    pub fn global(self: Module, name: [:0]const u8) Error!Global {
        var p: abi.Ptr = null;
        var len: usize = 0;
        try self.d.check(self.d.api.hipModuleGetGlobal(&p, &len, self.handle, name.ptr), "hipModuleGetGlobal");
        if (p == null or len == 0) return error.Invalid;
        return .{ .ptr = abi.addr(p), .len = len };
    }
};

pub const Function = struct {
    d: *const Driver,
    handle: abi.Function,

    pub fn attribute(self: Function, a: abi.FunctionAttribute) Error!c_int {
        var v: c_int = 0;
        try self.d.check(self.d.api.hipFuncGetAttribute(&v, a, self.handle), "hipFuncGetAttribute");
        return v;
    }

    /// Not available for module functions on HIP (hipFuncSetAttribute takes a runtime-API function pointer).
    pub fn setAttribute(_: Function, _: abi.FunctionAttribute, _: c_int) Error!void {
        return error.Unsupported;
    }

    /// Resident blocks a compute unit holds of this kernel at `threads` a block and `shared` dynamic bytes.
    pub fn occupancy(self: Function, threads: u32, shared: usize) Error!u32 {
        var n: c_int = 0;
        try self.d.check(self.d.api.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&n, self.handle, @intCast(threads), shared), "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
        return @intCast(@max(n, 0));
    }

    /// No opt-in exists on RDNA: a workgroup may use up to the device's LDS. Refuses more than that.
    pub fn allowDynamicShared(self: Function, bytes: u32) Error!void {
        const static: u32 = @intCast(try self.attribute(.shared_size_bytes));
        if (@as(u64, static) + bytes > 65536) return error.Unsupported;
    }
};

test "only a raw gfx1151 ELF passes as a gfx1151 image" {
    var img: [64]u8 align(8) = @splat(0);
    @memcpy(img[0..4], "\x7fELF");
    img[4] = 2;
    img[5] = 1;
    std.mem.writeInt(u16, img[18..20], em_amdgpu, .little);
    std.mem.writeInt(u32, img[48..52], 0x4a, .little);
    try checkImage(&img, "gfx1151");
    try std.testing.expectError(error.Unsupported, checkImage(&img, "gfx1150"));
    std.mem.writeInt(u32, img[48..52], 0x43, .little); // another gfx's code object
    try std.testing.expectError(error.NoKernelImage, checkImage(&img, "gfx1151"));
    const bundle = "__CLANG_OFFLOAD_BUNDLE__" ++ @as([40]u8, @splat(0));
    try std.testing.expectError(error.NoKernelImage, checkImage(bundle, "gfx1151"));
}
