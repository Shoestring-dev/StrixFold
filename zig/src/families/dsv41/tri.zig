//! Our Triton kernels from the captured set (aot.json + cubins), launched as the served build's wrappers launch them:
//! the same grid, constexprs and PDL. The set picks each launch's variant as Triton would (zig/src/cuda/aot.zig).
const std = @import("std");
const gpu = @import("gpu");
const aot = gpu.aot;
const tri_aux = @import("tri_aux.zig");

/// Where launches go: the captured set on a stream, or (tests) a log compared with the recorded launches.
/// kernels.decode_rows(): the most rows a call takes the row-invariant decode kernels with now: 16 (DECODE_ROWS), or a
/// concurrent round's row bound while round.forward runs one (kernels.round_rows(), Engine.round_rows).
pub var decode_rows: usize = 16;

pub const Tri = struct {
    set: ?*const aot.Set = null,
    stream: gpu.Stream = undefined,
    log: ?*Log = null,
    /// The "pdl" switch (TF_DS_TRITON_PDL, on in the served build): decode kernels as programmatic dependent launches.
    pdl: bool = true,
    /// HIP builds: w-tri's kernels for the launches tri_aux.zig names (null: every launch Triton).
    aux: ?*const tri_aux.Aux = null,

    pub fn run(t: Tri, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        if (t.log) |l| return l.add(name, grid, args, consts);
        try t.set.?.run(t.stream, name, grid, args, consts);
    }

    /// One of tri_aux.zig's kernels (`id` its kit id), its by-value argument struct `args`, blocks of 256 threads.
    pub fn runHip(t: Tri, id: []const u8, f: gpu.Function, grid: [3]u32, args: anytype) !void {
        return t.runHipBlock(id, f, grid, 256, args);
    }

    /// runHip for a kernel built for blocks of `block` threads (registry.json's block).
    pub fn runHipBlock(t: Tri, id: []const u8, f: gpu.Function, grid: [3]u32, block: u32, args: anytype) !void {
        if (t.log) |l| return l.addHip(id, grid, block, std.mem.asBytes(&args));
        var a: gpu.Args = .{};
        a.add(args);
        try gpu.launch.launch(f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = block } }, t.stream, &a);
    }

    /// The PDL constexpr a wrapper passes when its kernel takes one (_pdl() in kernels.py).
    pub fn pdlConst(t: Tri) aot.Const {
        return aot.ci("PDL", @intFromBool(t.pdl));
    }
};

pub fn u(x: usize) u32 {
    return @intCast(x);
}

pub fn cdiv(a: usize, b: usize) u32 {
    return u((a + b - 1) / b);
}

pub fn pow2(n: usize) usize {
    return std.math.ceilPowerOfTwo(usize, @max(n, 1)) catch unreachable;
}

/// A test log of launches: each one's function, grid, arguments and constexprs, copied.
pub const Log = struct {
    gpa: std.mem.Allocator,
    items: std.ArrayList(Kept) = .empty,

    /// `hip`: a HIP kernel's argument struct (its name the kit id, no Triton arguments or constexprs), `block` its threads
    pub const Kept = struct { name: []const u8, grid: [3]u32, args: []aot.Arg, consts: []aot.Const, hip: []const u8 = &.{}, block: u32 = 0 };

    fn add(l: *Log, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        try l.items.append(l.gpa, .{ .name = name, .grid = grid, .args = try l.gpa.dupe(aot.Arg, args), .consts = try l.gpa.dupe(aot.Const, consts) });
    }

    fn addHip(l: *Log, id: []const u8, grid: [3]u32, block: u32, bytes: []const u8) !void {
        try l.items.append(l.gpa, .{ .name = id, .grid = grid, .args = &.{}, .consts = &.{}, .hip = try l.gpa.dupe(u8, bytes), .block = block });
    }

    /// Launch `i`'s HIP argument struct, as `T`.
    pub fn hipArgs(l: *const Log, comptime T: type, i: usize) T {
        return std.mem.bytesToValue(T, l.items.items[i].hip[0..@sizeOf(T)]);
    }

    pub fn deinit(l: *Log) void {
        for (l.items.items) |k| {
            l.gpa.free(k.args);
            l.gpa.free(k.consts);
            l.gpa.free(k.hip);
        }
        l.items.deinit(l.gpa);
    }
};

// -- conformance with the recording (fixtures/launches.json: each Triton function's distinct recorded launches) -----

pub const fixtures = @embedFile("fixtures/launches.json");
/// The launches the recording lacks (tools/dsv41-hip/tri/variants.py: the prompt GEMM's shapes from the weights, every
/// decode / verify window size), held to the recording's standard: conform replays them through the wrappers too.
pub const derived_fixtures = @embedFile("fixtures/derived_launches.json");

/// One recorded launch: its tensors' dtype and shape (with 16-byte alignment), scalars and constexprs.
pub const Case = struct {
    v: std.json.ObjectMap,

    pub fn grid(c: Case) [3]u32 {
        const g = c.v.get("grid").?.array.items;
        return .{ @intCast(g[0].integer), @intCast(g[1].integer), @intCast(g[2].integer) };
    }

    /// A tensor argument's dimension `i`.
    pub fn dim(c: Case, tensor: []const u8, i: usize) usize {
        return @intCast(c.tensorV(tensor).array.items[1].array.items[i].integer);
    }

    /// A tensor argument's stride `i` (elements).
    pub fn stride(c: Case, tensor: []const u8, i: usize) usize {
        return @intCast(c.tensorV(tensor).array.items[2].array.items[i].integer);
    }

    pub fn rank(c: Case, tensor: []const u8) usize {
        return c.tensorV(tensor).array.items[1].array.items.len;
    }

    pub fn has(c: Case, tensor: []const u8) bool {
        return c.v.get("tensors").?.object.get(tensor) != null;
    }

    fn tensorV(c: Case, tensor: []const u8) std.json.Value {
        return c.v.get("tensors").?.object.get(tensor) orelse std.debug.panic("no tensor {s} in the case", .{tensor});
    }

    /// A fake device address for `tensor`: 16-aligned or not as recorded, distinct a name.
    pub fn ptr(c: Case, tensor: []const u8) u64 {
        const t = c.tensorV(tensor).array.items;
        const h: u64 = std.hash.Wyhash.hash(0, tensor) & 0xffff_ff00;
        return (0x7f00_0000_0000 + (h << 8)) + (if (t[3].bool) @as(u64, 0) else 2);
    }

    pub fn int(c: Case, name: []const u8) i64 {
        return c.v.get("scalars").?.object.get(name).?.integer;
    }

    pub fn float(c: Case, name: []const u8) f32 {
        const v = c.v.get("scalars").?.object.get(name).?;
        const bits = std.fmt.parseInt(u32, v.object.get("fp32_bits").?.string[2..], 16) catch unreachable;
        return @bitCast(bits);
    }

    pub fn constInt(c: Case, name: []const u8) i64 {
        const v = c.v.get("consts").?.object.get(name).?;
        return switch (v) {
            .integer => |i| i,
            .bool => |b| @intFromBool(b),
            else => std.debug.panic("constexpr {s} is not an integer", .{name}),
        };
    }

    pub fn hasConst(c: Case, name: []const u8) bool {
        return c.v.get("consts").?.object.get(name) != null;
    }
};

fn tritonType(dtype: []const u8) []const u8 {
    const names = .{ .{ "bfloat16", "*bf16" }, .{ "float16", "*fp16" }, .{ "float32", "*fp32" }, .{ "int64", "*i64" }, .{ "int32", "*i32" }, .{ "int16", "*i16" }, .{ "uint8", "*u8" }, .{ "int8", "*i8" }, .{ "bool", "*i1" }, .{ "float8_e4m3fn", "*fp8e4nv" } };
    inline for (names) |n| if (std.mem.eql(u8, dtype, n[0])) return n[1];
    return dtype;
}

/// Runs `call` on every recorded launch of `fn_name`, then on every derived one (derived_launches.json), and checks
/// that it asks for that launch: grid, every scalar (floats by their fp32 bits), every constexpr, and every tensor
/// argument's pointer type and alignment.
pub fn conform(fn_name: []const u8, call: *const fn (Tri, Case) anyerror!void) !void {
    const a = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, a, fixtures, .{});
    defer parsed.deinit();
    const cases = (parsed.value.object.get(fn_name) orelse return error.NoFixtures).array.items;
    try std.testing.expect(cases.len > 0);
    try conformCases(fn_name, call, cases);
    var derived = try std.json.parseFromSlice(std.json.Value, a, derived_fixtures, .{});
    defer derived.deinit();
    if (derived.value.object.get(fn_name)) |d| try conformCases(fn_name, call, d.array.items);
}

fn conformCases(fn_name: []const u8, call: *const fn (Tri, Case) anyerror!void, cases: []const std.json.Value) !void {
    const a = std.testing.allocator;
    for (cases, 0..) |cv, ci| {
        const c: Case = .{ .v = cv.object };
        var log: Log = .{ .gpa = a };
        defer log.deinit();
        try call(.{ .log = &log, .pdl = !c.hasConst("PDL") or c.constInt("PDL") != 0 }, c);
        const k = for (log.items.items) |k| {
            if (std.mem.eql(u8, k.name, fn_name)) break k;
        } else {
            std.debug.print("case {d}: the wrapper launched no {s}\n", .{ ci, fn_name });
            return error.TestUnexpectedResult;
        };
        errdefer std.debug.print("{s} case {d} (phase {s})\n", .{ fn_name, ci, c.v.get("phase").?.string });
        try std.testing.expectEqual(c.grid(), k.grid);
        var consts = c.v.get("consts").?.object.iterator();
        recorded: while (consts.next()) |e| {
            const got = for (k.consts) |x| {
                if (std.mem.eql(u8, x.name, e.key_ptr.*)) break x;
            } else {
                if (e.value_ptr.* == .integer and e.value_ptr.integer == 1 and folded(k.args, e.key_ptr.*)) continue :recorded;
                std.debug.print("constexpr {s} not passed\n", .{e.key_ptr.*});
                return error.TestUnexpectedResult;
            };
            switch (e.value_ptr.*) {
                .integer => |i| try std.testing.expectEqual(@as(?i64, i), got.int),
                .bool => |b| try std.testing.expectEqual(@as(?i64, @intFromBool(b)), got.int),
                .object => |o| try std.testing.expectEqual(std.fmt.parseInt(u32, o.get("fp32_bits").?.string[2..], 16) catch unreachable, @as(u32, @bitCast(got.f32.?))),
                else => return error.UnsupportedConstexpr,
            }
        }
        for (k.consts) |x| if (c.v.get("consts").?.object.get(x.name) == null) {
            std.debug.print("constexpr {s} passed but not in the variant\n", .{x.name});
            return error.TestUnexpectedResult;
        };
        var scalars = c.v.get("scalars").?.object.iterator();
        while (scalars.next()) |e| {
            const got = for (k.args) |x| {
                if (std.mem.eql(u8, x.name, e.key_ptr.*)) break x;
            } else {
                std.debug.print("scalar {s} not passed\n", .{e.key_ptr.*});
                return error.TestUnexpectedResult;
            };
            switch (e.value_ptr.*) {
                .integer => |i| switch (got.value) {
                    .i32 => |x| try std.testing.expectEqual(i, @as(i64, x)),
                    .u64 => |x| try std.testing.expectEqual(i, @as(i64, @intCast(x))),
                    else => return error.TestUnexpectedResult,
                },
                .object => |o| try std.testing.expectEqual(std.fmt.parseInt(u32, o.get("fp32_bits").?.string[2..], 16) catch unreachable, @as(u32, @bitCast(got.value.f32))),
                else => return error.UnsupportedScalar,
            }
        }
        var tensors = c.v.get("tensors").?.object.iterator();
        while (tensors.next()) |e| {
            const got = for (k.args) |x| {
                if (std.mem.eql(u8, x.name, e.key_ptr.*)) break x;
            } else {
                std.debug.print("tensor {s} not passed\n", .{e.key_ptr.*});
                return error.TestUnexpectedResult;
            };
            const t = e.value_ptr.array.items;
            try std.testing.expectEqualStrings(tritonType(t[0].string), got.value.ptr.ty);
            try std.testing.expectEqual(t[3].bool, got.value.ptr.addr % 16 == 0);
        }
        // and no argument the kernel does not take, which aot.Set.find would refuse
        for (k.args) |x| {
            if (c.v.get("scalars").?.object.get(x.name) != null or c.v.get("tensors").?.object.get(x.name) != null) continue;
            if (folded(k.args, x.name)) {
                if (c.v.get("consts").?.object.get(x.name)) |r| if (r == .integer and r.integer == 1) continue;
            }
            std.debug.print("argument {s} passed but not taken by this variant\n", .{x.name});
            return error.TestUnexpectedResult;
        }
    }
}

/// Whether the recorded or derived launches (the kit's variant list: t_cover checks the kit holds each one) hold
/// `k`'s specialization, the strict-mode question asked of the fixtures: its constexprs, each pointer's type and
/// 16-byte alignment, each int's Triton class (1 folded into a constexpr, a multiple of 16, other; for kernels
/// without do_not_specialize ints), each float present.
pub fn covered(gpa: std.mem.Allocator, k: Log.Kept) !bool {
    for ([_][]const u8{ fixtures, derived_fixtures }) |src| {
        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, src, .{});
        defer parsed.deinit();
        const cases = (parsed.value.object.get(k.name) orelse continue).array.items;
        for (cases) |cv| if (sameVariant(.{ .v = cv.object }, k)) return true;
    }
    return false;
}

fn sameVariant(c: Case, k: Log.Kept) bool {
    const consts = c.v.get("consts").?.object;
    for (k.consts) |x| {
        const r = consts.get(x.name) orelse return false;
        switch (r) {
            .integer => |i| if (x.int == null or x.int.? != i) return false,
            .bool => |b| if (x.int == null or x.int.? != @intFromBool(b)) return false,
            .object => |o| {
                const bits = std.fmt.parseInt(u32, o.get("fp32_bits").?.string[2..], 16) catch return false;
                if (x.f32 == null or @as(u32, @bitCast(x.f32.?)) != bits) return false;
            },
            else => return false,
        }
    }
    var it = consts.iterator();
    while (it.next()) |e| {
        const passed = for (k.consts) |x| {
            if (std.mem.eql(u8, x.name, e.key_ptr.*)) break true;
        } else false;
        if (!passed and !(e.value_ptr.* == .integer and e.value_ptr.integer == 1 and folded(k.args, e.key_ptr.*))) return false;
    }
    const scalars = c.v.get("scalars").?.object;
    const tensors = c.v.get("tensors").?.object;
    for (k.args) |a| switch (a.value) {
        .ptr => |pv| {
            const t = (tensors.get(a.name) orelse return false).array.items;
            if (!std.mem.eql(u8, tritonType(t[0].string), pv.ty) or t[3].bool != (pv.addr % 16 == 0)) return false;
        },
        .i32 => |x| {
            if (x == 1) {
                const r = consts.get(a.name) orelse return false;
                if (!(r == .integer and r.integer == 1)) return false;
                continue;
            }
            const r = scalars.get(a.name) orelse return false;
            if (r != .integer or r.integer == 1 or (@mod(r.integer, 16) == 0) != (@mod(x, 16) == 0)) return false;
        },
        .f32, .u64 => if (scalars.get(a.name) == null) return false,
    };
    return true;
}

/// An int argument of 1, which Triton folds into a constexpr of the variant; aot.Set.find takes either form.
fn folded(args: []const aot.Arg, name: []const u8) bool {
    for (args) |a| if (std.mem.eql(u8, a.name, name)) return a.value == .i32 and a.value.i32 == 1;
    return false;
}
