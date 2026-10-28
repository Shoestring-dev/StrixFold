//! Triton kernels compiled ahead of time for the device (kit entries with abi "triton"): each launch picks the variant
//! Triton itself would have picked, by the rules of zig/src/cuda/aot.zig (exact constexprs, 16-divisibility and
//! ==1 specialization per argument), with the same Arg/Const API so the family's tri*.zig is unchanged. Launch geometry
//! comes from the kit: a block of num_warps waves of the kernel's own wavefront width, its metadata's LDS bytes.
//! Strict mode (the kit's) refuses the "assumes less" fallback, so a qualification run proves every launch had its own
//! variant.

const std = @import("std");
const Stream = @import("stream.zig").Stream;
const Function = @import("module.zig").Function;
const launch = @import("launch.zig");
const kit_mod = @import("kit.zig");
const manifest = @import("../manifest.zig");

/// One argument of a launch, by the kernel's parameter name.
pub const Arg = struct {
    name: []const u8,
    value: Value,

    pub const Value = union(enum) { ptr: struct { addr: u64, ty: []const u8 }, i32: i32, f32: f32, u64: u64 };
};

/// A constexpr the call site compiled the kernel with: ints, bools, fp32 bits.
pub const Const = struct { name: []const u8, int: ?i64 = null, f32: ?f32 = null };

pub fn ptr(name: []const u8, ty: []const u8, addr: u64) Arg {
    return .{ .name = name, .value = .{ .ptr = .{ .addr = addr, .ty = ty } } };
}
pub fn int(name: []const u8, v: i32) Arg {
    return .{ .name = name, .value = .{ .i32 = v } };
}
pub fn float(name: []const u8, v: f32) Arg {
    return .{ .name = name, .value = .{ .f32 = v } };
}
pub fn word(name: []const u8, v: u64) Arg {
    return .{ .name = name, .value = .{ .u64 = v } };
}
pub fn ci(name: []const u8, v: i64) Const {
    return .{ .name = name, .int = v };
}
pub fn cf(name: []const u8, v: f32) Const {
    return .{ .name = name, .f32 = v };
}

const Variant = struct {
    t: manifest.Triton,
    f: Function,
    entry: u32,
    wave: u32,
    told: std.atomic.Value(bool) = .init(false),
};

pub const Set = struct {
    gpa: std.mem.Allocator,
    variants: []Variant,
    /// Refuse fallbacks (the kit's strict mode).
    strict: bool,
    /// Launches that took a fallback variant (non-strict runs).
    fallbacks: std.atomic.Value(u64) = .init(0),

    /// Every Triton entry of `k`; each one's parameter list (plus its hidden scratch pointers) must have the code object's
    /// explicit-argument count, and its scratch must be zero (Triton's global/profile scratch is not provided).
    pub fn fromKit(gpa: std.mem.Allocator, k: *const kit_mod.Kit) !Set {
        var list: std.ArrayList(Variant) = .empty;
        errdefer list.deinit(gpa);
        for (k.entries, 0..) |e, i| {
            const t = e.k.triton orelse continue;
            if (!std.mem.eql(u8, e.k.abi, "triton")) return error.KitManifest;
            if (t.params.len + t.hidden.len != e.explicit.len) {
                std.log.err("{s}: Triton lists {d} + {d} hidden parameters, the code object {d}", .{ e.k.id, t.params.len, t.hidden.len, e.explicit.len });
                return error.AbiMismatch;
            }
            for (t.params, e.explicit[0..t.params.len]) |p, a| {
                if (a.size != paramSize(p.type)) {
                    std.log.err("{s}: parameter {s} ({s}) is {d} bytes in the code object", .{ e.k.id, p.name, p.type, a.size });
                    return error.AbiMismatch;
                }
            }
            try list.append(gpa, .{ .t = t, .f = e.f, .entry = @intCast(i), .wave = e.k.wave });
        }
        return .{ .gpa = gpa, .variants = try list.toOwnedSlice(gpa), .strict = k.strict };
    }

    pub fn deinit(self: *Set) void {
        self.gpa.free(self.variants);
        self.* = undefined;
    }

    /// The variant of `function` compiled for these constexprs and these arguments' specialization; without one, in a
    /// non-strict run, the fitting variant that assumes least less (named once on stderr).
    pub fn find(self: *const Set, function: []const u8, args: []const Arg, consts: []const Const) !*Variant {
        for (self.variants) |*v| {
            if (!std.mem.eql(u8, v.t.@"fn", function)) continue;
            if (matches(v.t, args, consts)) return v;
        }
        if (!self.strict) {
            var best: ?*Variant = null;
            var best_held: usize = 0;
            for (self.variants) |*v| {
                if (!std.mem.eql(u8, v.t.@"fn", function)) continue;
                const held = fits(v.t, args, consts) orelse continue;
                if (best == null or held > best_held) {
                    best = v;
                    best_held = held;
                }
            }
            if (best) |v| {
                if (!v.told.swap(true, .monotonic)) std.log.warn("{s}: a launch whose specialization the kit lacks runs a variant that assumes less", .{function});
                _ = @constCast(&self.fallbacks).fetchAdd(1, .monotonic);
                return v;
            }
        }
        std.log.err("no {s}Triton variant of {s} in the kit for this launch:", .{ if (self.strict) "exact (strict) " else "", function });
        for (args) |a| switch (a.value) {
            .ptr => |p| std.log.err("  {s}: {s} at {x} (16-aligned {})", .{ a.name, p.ty, p.addr, p.addr % 16 == 0 }),
            .i32 => |x| std.log.err("  {s}: i32 {d}", .{ a.name, x }),
            .f32 => |x| std.log.err("  {s}: fp32 {d}", .{ a.name, x }),
            .u64 => |x| std.log.err("  {s}: u64 {d}", .{ a.name, x }),
        };
        for (consts) |c| std.log.err("  constexpr {s} = {?d} {?d}", .{ c.name, c.int, c.f32 });
        return error.MissingTritonVariant;
    }

    /// The smallest value of constexpr `name` at or above `at_least` among `function`'s variants.
    pub fn smallestConst(self: *const Set, function: []const u8, name: []const u8, at_least: i64) ?i64 {
        var best: ?i64 = null;
        for (self.variants) |v| {
            if (!std.mem.eql(u8, v.t.@"fn", function)) continue;
            const c = (v.t.consts.map.get(name) orelse continue).int orelse continue;
            if (c >= at_least and (best == null or c < best.?)) best = c;
        }
        return best;
    }

    /// Launches `function` on `grid` as Triton's launcher would: runtime arguments in the variant's order, then a null
    /// for each hidden scratch pointer; block = num_warps * the kernel's wavefront width.
    pub fn run(self: *const Set, stream: Stream, function: []const u8, grid: [3]u32, args: []const Arg, consts: []const Const) !void {
        const v = try self.find(function, args, consts);
        var packed_args: launch.Args = .{};
        for (v.t.params) |p| {
            const a = lookup(args, p.name).?;
            switch (a.value) {
                .ptr => |x| packed_args.add(x.addr),
                .i32 => |x| packed_args.add(x),
                .f32 => |x| packed_args.add(x),
                .u64 => |x| packed_args.add(x),
            }
        }
        for (v.t.hidden) |_| packed_args.add(@as(u64, 0));
        try launch.launch(v.f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = v.t.num_warps * v.wave }, .shared = v.t.shared }, stream, &packed_args);
    }
};

const paramSize = manifest.tritonParamSize;

fn lookup(args: []const Arg, name: []const u8) ?Arg {
    for (args) |a| if (std.mem.eql(u8, a.name, name)) return a;
    return null;
}

/// Whether variant `k` may run this launch though compiled for a narrower one (zig/src/cuda/aot.zig's rule): the same
/// constexprs, and every assumption it makes held by the launch. Null when it may not, else how many of the launch's own
/// specializations it shares.
fn fits(k: manifest.Triton, args: []const Arg, consts: []const Const) ?usize {
    for (consts) |c| {
        const got = k.consts.map.get(c.name) orelse return null;
        if (c.int) |x| if (got.int == null or got.int.? != x) return null;
        if (c.f32) |x| if (got.f32 == null or got.f32.? != @as(u32, @bitCast(x))) return null;
    }
    var held: usize = 0;
    var runtime: usize = 0;
    for (args) |a| {
        const param = for (k.params) |p| {
            if (std.mem.eql(u8, p.name, a.name)) break p;
        } else null;
        switch (a.value) {
            .i32 => |x| {
                if (param == null) {
                    const got = k.consts.map.get(a.name) orelse return null;
                    if (x != 1 or got.int == null or got.int.? != 1) return null;
                    held += 1;
                    continue;
                }
                const p = param.?;
                if (!std.mem.eql(u8, p.type, "i32")) return null;
                const d16 = !p.nospec and x != 1 and @mod(x, 16) == 0;
                if (p.div16 and !d16) return null;
                if (p.div16 == d16 and (p.nospec or x != 1)) held += 1;
            },
            .ptr => |x| {
                const p = param orelse return null;
                if (!std.mem.eql(u8, p.type, x.ty)) return null;
                const d16 = x.addr % 16 == 0;
                if (p.div16 and !d16) return null;
                if (p.div16 == d16) held += 1;
            },
            .f32 => {
                const p = param orelse return null;
                if (!std.mem.eql(u8, p.type, "fp32")) return null;
                held += 1;
            },
            .u64 => |x| {
                const p = param orelse return null;
                if (!std.mem.eql(u8, p.type, "u64")) return null;
                const d16 = x % 16 == 0;
                if (p.div16 and !d16) return null;
                if (p.div16 == d16) held += 1;
            },
        }
        runtime += 1;
    }
    if (runtime != k.params.len) return null;
    return held;
}

fn matches(k: manifest.Triton, args: []const Arg, consts: []const Const) bool {
    for (consts) |c| {
        const got = k.consts.map.get(c.name) orelse return false;
        if (c.int) |x| if (got.int == null or got.int.? != x) return false;
        if (c.f32) |x| if (got.f32 == null or got.f32.? != @as(u32, @bitCast(x))) return false;
    }
    var runtime: usize = 0;
    for (args) |a| {
        const param = for (k.params) |p| {
            if (std.mem.eql(u8, p.name, a.name)) break p;
        } else null;
        switch (a.value) {
            .i32 => |x| {
                if (param == null) {
                    // an int Triton folded: only the value 1 is ever specialized to a constexpr
                    const got = k.consts.map.get(a.name) orelse return false;
                    if (x != 1 or got.int == null or got.int.? != 1) return false;
                    continue;
                }
                const p = param.?;
                if (!std.mem.eql(u8, p.type, "i32")) return false;
                if (!p.nospec and x == 1) return false;
                if (p.div16 != (!p.nospec and @mod(x, 16) == 0)) return false;
            },
            .ptr => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, x.ty) or p.div16 != (x.addr % 16 == 0)) return false;
            },
            .f32 => {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "fp32")) return false;
            },
            .u64 => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "u64") or p.div16 != (x % 16 == 0)) return false;
            },
        }
        runtime += 1;
    }
    return runtime == k.params.len;
}

test "variant selection follows Triton's specialization (the CUDA set's rules)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var consts: std.json.ArrayHashMap(manifest.TritonConst) = .{};
    try consts.map.put(a, "D", .{ .int = 4096 });
    const exact: manifest.Triton = .{ .@"fn" = "_k", .params = &.{ .{ .name = "X", .type = "*bf16", .div16 = true }, .{ .name = "n", .type = "i32", .div16 = true } }, .consts = consts, .num_warps = 4 };
    const general: manifest.Triton = .{ .@"fn" = "_k", .params = &.{ .{ .name = "X", .type = "*bf16" }, .{ .name = "n", .type = "i32" } }, .consts = consts, .num_warps = 4 };
    const aligned = [_]Arg{ ptr("X", "*bf16", 0x1000), int("n", 64) };
    const odd = [_]Arg{ ptr("X", "*bf16", 0x1002), int("n", 63) };
    const cs = [_]Const{ci("D", 4096)};
    try std.testing.expect(matches(exact, &aligned, &cs));
    try std.testing.expect(!matches(exact, &odd, &cs));
    try std.testing.expect(matches(general, &odd, &cs));
    try std.testing.expect(!matches(general, &aligned, &cs));
    // the general variant fits the aligned launch (assuming less), the exact one never fits the odd launch
    try std.testing.expect(fits(general, &aligned, &cs) != null);
    try std.testing.expect(fits(exact, &odd, &cs) == null);
    try std.testing.expect(!matches(exact, &aligned, &.{ci("D", 2048)}));
    // an int of value 1 is a constexpr to Triton
    try std.testing.expect(!matches(general, &.{ ptr("X", "*bf16", 0x1002), int("n", 1) }, &cs));
}
