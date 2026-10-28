//! One HIP device made current on the calling thread. HIP binds a device per thread (hipSetDevice); the deprecated
//! hipCtx* calls are not used. The same surface as the CUDA runtime's Context, so the family's calls are unchanged.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const caps_mod = @import("caps.zig");

const Properties = @import("driver.zig").Properties;
extern fn tf_hip_device_arch(Properties, c_int, [*]u8, usize) c_int;

pub const Context = struct {
    d: *const Driver,
    device: c_int,

    pub fn init(d: *const Driver, ordinal: c_int) Error!Context {
        if (ordinal < 0) return error.Invalid;
        const n = try d.deviceCount();
        if (ordinal >= n) return error.Invalid;
        try d.check(d.api.hipSetDevice(ordinal), "hipSetDevice");
        return .{ .d = d, .device = ordinal };
    }

    /// Waits for all work on the device; resources made in it must be released first.
    pub fn deinit(self: *Context) void {
        _ = self.d.api.hipDeviceSynchronize();
        self.* = undefined;
    }

    /// Makes the device current on another thread before it calls the runtime.
    pub fn makeCurrent(self: *const Context) Error!void {
        try self.d.check(self.d.api.hipSetDevice(self.device), "hipSetDevice");
    }

    pub fn synchronize(self: *const Context) Error!void {
        try self.d.check(self.d.api.hipDeviceSynchronize(), "hipDeviceSynchronize");
    }

    pub fn attribute(self: *const Context, a: abi.DeviceAttribute) Error!c_int {
        var v: c_int = 0;
        try self.d.check(self.d.api.hipDeviceGetAttribute(&v, a, self.device), "hipDeviceGetAttribute");
        return v;
    }

    pub fn name(self: *const Context, buf: []u8) Error![]const u8 {
        if (buf.len < 2) return error.Invalid;
        try self.d.check(self.d.api.hipDeviceGetName(buf.ptr, @intCast(buf.len), self.device), "hipDeviceGetName");
        return std.mem.sliceTo(buf, 0);
    }

    /// The device's exact gfx name with its feature suffixes ("gfx1151", or "gfx1151:xnack-" on some stacks).
    pub fn arch(self: *const Context, buf: *[256]u8) Error![]const u8 {
        const r = tf_hip_device_arch(self.d.properties, self.device, buf, buf.len);
        if (r != 0) {
            if (r > 0) try self.d.check(r, "hipGetDevicePropertiesR0600");
            return error.Invalid;
        }
        return std.mem.sliceTo(buf, 0);
    }

    /// The device facts a family reads (section 3.2 of tools/dsv41-hip/ADAPTER.md); a GPU outside the ISA table is
    /// refused rather than treated as a neighbour.
    pub fn caps(self: *const Context) Error!caps_mod.Caps {
        var buf: [256]u8 = undefined;
        const full = try self.arch(&buf);
        const base = full[0 .. std.mem.indexOfScalar(u8, full, ':') orelse full.len];
        const isa = caps_mod.Isa.of(base) orelse return error.Unsupported;
        const wave = try self.attribute(.warp_size);
        if (wave != isa.wave) return error.Unsupported;
        var c: caps_mod.Caps = .{
            .isa = isa,
            .cus = @intCast(try self.attribute(.multiprocessor_count)),
            .lds_per_block = @intCast(try self.attribute(.max_shared_memory_per_block)),
            .integrated = (try self.attribute(.integrated)) != 0,
            // hipDeviceAttributeWallClockRate is in kHz
            .wall_clock_hz = @as(u64, @intCast(try self.attribute(.wall_clock_rate))) * 1000,
        };
        if (base.len > c.arch.len) return error.Invalid;
        @memcpy(c.arch[0..base.len], base);
        c.arch_len = @intCast(base.len);
        return c;
    }

    pub const MemInfo = struct { free: usize, total: usize };

    pub fn memInfo(self: *const Context) Error!MemInfo {
        var m: MemInfo = .{ .free = 0, .total = 0 };
        try self.d.check(self.d.api.hipMemGetInfo(&m.free, &m.total), "hipMemGetInfo");
        if (m.total == 0 or m.free > m.total) return error.Invalid;
        return m;
    }
};
