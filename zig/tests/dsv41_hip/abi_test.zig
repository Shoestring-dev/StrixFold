//! tf-dsv41-hip-abi-test --kit DIR --scratch DIR [--extra-kit DIR] [--receipt PATH]: the kernel ABI end to end on the GPU (ADAPTER.md 6.6).
//! Loads the test kit through the manifest loader, binds every echo kernel with its Zig parameter declaration, fills
//! every field with a distinct pattern, launches through the family's launch path and compares every byte the device
//! read. Also drives the Triton variant set, and checks the refusals: a wrong declaration, a tampered object, another ABI
//! version, a missing required kernel, a launch whose packing or block disagrees with the metadata (strict mode), and a
//! Triton launch with no exact variant. --extra-kit: another kit (w-tri's imported Triton kit) opened through the same
//! loader: every object loaded, every variant resolved and bound into the Triton set. Run under the GPU lock
//! (tools/dsv41-hip/run_gpu_tests.sh).

const std = @import("std");
const gpu = @import("gpu");
const abi_h = gpu.kernel_abi;

const Fail = error{CheckFailed};

fn errIs(x: anytype, want: anyerror) bool {
    if (x) |_| return false else |e| return e == want;
}

const Receipt = struct {
    gpa: std.mem.Allocator,
    checks: std.ArrayList(u8) = .empty,
    passed: u32 = 0,
    failed: u32 = 0,
    detail: [768]u8 = undefined,
    detail_len: usize = 0,

    fn note(r: *Receipt, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(r.detail[r.detail_len..], fmt, args) catch return;
        r.detail_len += s.len;
    }

    fn expect(r: *Receipt, ok: bool, comptime fmt: []const u8, args: anytype) Fail!void {
        if (ok) return;
        r.note("FAILED: " ++ fmt, args);
        return error.CheckFailed;
    }

    fn run(r: *Receipt, name: []const u8, comptime f: anytype, ctx: anytype) !void {
        r.detail_len = 0;
        const ok = if (f(r, ctx)) |_| true else |e| blk: {
            if (e != error.CheckFailed) r.note(" error {s}", .{@errorName(e)});
            break :blk false;
        };
        if (ok) r.passed += 1 else r.failed += 1;
        std.debug.print("{s} {s}: {s}\n", .{ if (ok) "PASS" else "FAIL", name, r.detail[0..r.detail_len] });
        const sep: []const u8 = if (r.checks.items.len == 0) "" else ",\n";
        try r.checks.appendSlice(r.gpa, try std.fmt.allocPrint(r.gpa, "{s}    {{\"name\": \"{s}\", \"ok\": {}, \"detail\": \"", .{ sep, name, ok }));
        for (r.detail[0..r.detail_len]) |c| {
            if (c == '"' or c == '\\') try r.checks.append(r.gpa, '\\');
            try r.checks.append(r.gpa, if (c < 0x20) ' ' else c);
        }
        try r.checks.appendSlice(r.gpa, "\"}");
    }
};

const Env = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const gpu.Driver,
    arch: []const u8,
    kit: *gpu.Kit,
    kit_dir: []const u8,
    scratch: []const u8,
    s: gpu.Stream,
    out: gpu.DeviceBuffer,
};

/// splitmix64: distinct, address-like patterns.
fn mix(x0: u64) u64 {
    var x = x0 +% 0x9E3779B97F4A7C15;
    x = (x ^ (x >> 30)) *% 0xBF58476D1CE4E5B9;
    x = (x ^ (x >> 27)) *% 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

const EchoMixed = gpu.Kernel("abi.echo_mixed", extern struct { a: abi_h.tf_ds_abi_mixed, out: u64 });
const EchoBig = gpu.Kernel("abi.echo_big", extern struct { a: abi_h.tf_ds_abi_big, out: u64 });
const EchoParams = gpu.Kernel("abi.echo_params", extern struct { a: u8, b: u64, c: u16, d: f32, e: i8, f: f64, g: i32, out: u64 });
/// The same kernel declared with one parameter one byte wide too narrow.
const EchoParamsWrong = gpu.Kernel("abi.echo_params", extern struct { a: u8, b: u64, c: u8, d: f32, e: i8, f: f64, g: i32, out: u64 });

fn structs(r: *Receipt, e: *const Env) !void {
    const k = try EchoMixed.bind(e.kit, &.{});
    var m: abi_h.tf_ds_abi_mixed = undefined;
    m.addr = 0x0000_7f12_3456_9abc;
    m.i32 = -123456789;
    m.f32 = @bitCast(@as(u32, 0x7fc12345)); // a NaN with a payload: copied, never computed with
    m.words = .{ mix(1), mix(2), mix(3) };
    m.i64 = -0x0123456789abcdef;
    m.u16 = 0xbeef;
    m.u8 = 0x5a;
    m.pad0 = 0xa5;
    m.u32 = 0xfedcba98;
    m.f64 = @bitCast(@as(u64, 0x8000000000000001)); // a negative denormal
    try e.out.fill8(0, null);
    try k.launch(e.s, .{ .grid = .{ .x = 1 }, .block = .{ .x = 64 } }, .{ .a = m, .out = e.out.ptr });
    try e.s.synchronize();
    var got: [@sizeOf(abi_h.tf_ds_abi_mixed)]u8 = undefined;
    try e.out.download(0, &got);
    const want = std.mem.asBytes(&m);
    for (want, got, 0..) |w, g, i| try r.expect(w == g, "tf_ds_abi_mixed byte {d}: sent {x}, device read {x}", .{ i, w, g });
    const kb = try EchoBig.bind(e.kit, &.{});
    var big: abi_h.tf_ds_abi_big = undefined;
    for (&big.w, 0..) |*x, i| x.* = mix(100 + i);
    big.n = 7;
    big.m = -9;
    big.discard = 1;
    big.pad0 = 0x5555aaaa;
    try e.out.fill8(0, null);
    try kb.launch(e.s, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, .{ .a = big, .out = e.out.ptr });
    try e.s.synchronize();
    var gotb: [@sizeOf(abi_h.tf_ds_abi_big)]u8 = undefined;
    try e.out.download(0, &gotb);
    const wantb = std.mem.asBytes(&big);
    for (wantb, gotb, 0..) |w, g, i| try r.expect(w == g, "tf_ds_abi_big byte {d}: sent {x}, device read {x}", .{ i, w, g });
    r.note("by-value structs from abi.h (one definition for hipcc and Zig): {d} and {d} bytes, every byte as sent (NaN payload, negative denormal, padding byte, 1424-byte kernarg)", .{ want.len, wantb.len });
}

const EchoGArgs = gpu.Kernel("abi.echo_exl3_gargs", extern struct { a: abi_h.tf_ds_exl3_gargs, out: u64 });
const EchoRArgs = gpu.Kernel("abi.echo_exl3_rargs", extern struct { a: abi_h.tf_ds_exl3_rargs, out: u64 });
/// exl3.experts.grouped_cp's list (hip_load.params.GroupedCp): ten addresses, six ints, the decode epilogue; then out.
const EchoCp = gpu.Kernel("abi.echo_exl3_cp", extern struct {
    xa: u64, xb: u64, wa: u64, wb: u64, ka: u64, kb: u64, ids: u64, count: u64, members: u64, z: u64,
    k: i32, n: i32, p: i32, splits: i32, maxm: i32, slots: i32, epi: abi_h.tf_ds_exl3_decode_epi, out: u64,
});

/// w-exl3's by-value structs: every byte of glinear's 1,424-byte and rot_many's 392-byte arguments, and grouped_cp's list
/// with its 144-byte epilogue after six ints, as the device received them.
fn exl3Structs(r: *Receipt, e: *const Env) !void {
    var g: abi_h.tf_ds_exl3_gargs = undefined;
    const gb = std.mem.asBytes(&g);
    for (gb, 0..) |*b, i| b.* = @truncate(mix(1000 + i));
    var rg: abi_h.tf_ds_exl3_rargs = undefined;
    const rb = std.mem.asBytes(&rg);
    for (rb, 0..) |*b, i| b.* = @truncate(mix(5000 + i));
    var cp: EchoCp.Params = undefined;
    const cb = std.mem.asBytes(&cp)[0..@offsetOf(EchoCp.Params, "out")];
    for (cb, 0..) |*b, i| b.* = @truncate(mix(9000 + i));
    try r.expect(cb.len == 248 and @offsetOf(EchoCp.Params, "epi") == 104, "GroupedCp list layout", .{});
    const kg = try EchoGArgs.bind(e.kit, &.{});
    try e.out.fill8(0, null);
    try kg.launch(e.s, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, .{ .a = g, .out = e.out.ptr });
    try e.s.synchronize();
    var got: [@sizeOf(abi_h.tf_ds_exl3_gargs)]u8 = undefined;
    try e.out.download(0, &got);
    for (gb, got, 0..) |w, x, i| try r.expect(w == x, "tf_ds_exl3_gargs byte {d}: sent {x}, device read {x}", .{ i, w, x });
    const kr = try EchoRArgs.bind(e.kit, &.{});
    try e.out.fill8(0, null);
    try kr.launch(e.s, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, .{ .a = rg, .out = e.out.ptr });
    try e.s.synchronize();
    var gotr: [@sizeOf(abi_h.tf_ds_exl3_rargs)]u8 = undefined;
    try e.out.download(0, &gotr);
    for (rb, gotr, 0..) |w, x, i| try r.expect(w == x, "tf_ds_exl3_rargs byte {d}: sent {x}, device read {x}", .{ i, w, x });
    const kc = try EchoCp.bind(e.kit, &.{});
    try e.out.fill8(0, null);
    cp.out = e.out.ptr;
    try kc.launch(e.s, .{ .grid = .{ .x = 1 }, .block = .{ .x = 64 } }, cp);
    try e.s.synchronize();
    var gotc: [248]u8 = undefined;
    try e.out.download(0, &gotc);
    for (cb, gotc, 0..) |w, x, i| try r.expect(w == x, "grouped_cp list byte {d}: sent {x}, device read {x}", .{ i, w, x });
    r.note("EXL3 by-value arguments (abi.h): tf_ds_exl3_gargs {d} B, tf_ds_exl3_rargs {d} B, grouped_cp's list {d} B (epilogue at 104), every byte as sent", .{ gb.len, rb.len, cb.len });
}

fn params(r: *Receipt, e: *const Env) !void {
    const k = try EchoParams.bind(e.kit, &.{});
    const P = EchoParams.Params;
    const p: P = .{ .a = 0xc3, .b = 0x8123_4567_89ab_cdef, .c = 0x7e57, .d = -1.5e-30, .e = -77, .f = 3.0e300, .g = -2147483000, .out = e.out.ptr };
    try e.out.fill8(0, null);
    try k.launch(e.s, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, p);
    try e.s.synchronize();
    var o: [7]u64 = undefined;
    try e.out.download(0, std.mem.sliceAsBytes(&o));
    const want = [7]u64{ p.a, p.b, p.c, @as(u32, @bitCast(p.d)), @bitCast(@as(i64, p.e)), @bitCast(p.f), @bitCast(@as(i64, p.g)) };
    for (want, o, 0..) |w, g, i| try r.expect(w == g, "parameter {d}: sent {x}, device read {x}", .{ i, w, g });
    const t = EchoParams.params;
    r.note("u8,u64,u16,f32,i8,f64,i32,u64 at kernarg offsets", .{});
    for (t) |x| r.note(" {d}", .{x.offset});
    r.note(" (padding between each), every value as sent", .{});
    const wrong = EchoParamsWrong.bind(e.kit, &.{});
    r.note("; a declaration with c: u8 instead of u16 -> {s}", .{if (wrong) |_| "bound?!" else |x| @errorName(x)});
    try r.expect(errIs(wrong, error.AbiMismatch), "wrong declaration was bound", .{});
}

fn triton(r: *Receipt, e: *const Env) !void {
    var set = try gpu.aot.Set.fromKit(e.gpa, e.kit);
    defer set.deinit();
    var x = try gpu.DeviceBuffer.alloc(e.d, 64);
    defer x.free();
    const aot = gpu.aot;
    const consts = [_]aot.Const{aot.ci("BLOCK", 32)};
    var o: [7]u64 = undefined;
    // aligned X and n % 16 == 0: the specialized variant A
    try set.run(e.s, "_echo", .{ 1, 1, 1 }, &.{ aot.ptr("X", "*fp32", x.ptr), aot.int("n", 64), aot.float("s", 2.5), aot.ptr("OUT", "*u64", e.out.ptr) }, &consts);
    try e.s.synchronize();
    try e.out.download(0, std.mem.sliceAsBytes(&o));
    try r.expect(o[0] == 0xA and o[1] == x.ptr and o[2] == 64 and o[3] == @as(u32, @bitCast(@as(f32, 2.5))) and o[4] == 0 and o[5] == 0 and o[6] == 64, "variant A saw {x} {x} {d} {x} {x} {x} block {d}", .{ o[0], o[1], o[2], o[3], o[4], o[5], o[6] });
    // misaligned X and odd n: the general variant B
    try set.run(e.s, "_echo", .{ 1, 1, 1 }, &.{ aot.ptr("X", "*fp32", x.ptr + 4), aot.int("n", 63), aot.float("s", -1.0), aot.ptr("OUT", "*u64", e.out.ptr) }, &consts);
    try e.s.synchronize();
    try e.out.download(0, std.mem.sliceAsBytes(&o));
    try r.expect(o[0] == 0xB and o[1] == x.ptr + 4 and o[2] == 63 and o[6] == 32, "variant B saw {x} {x} {d} block {d}", .{ o[0], o[1], o[2], o[6] });
    // aligned X with odd n: no exact variant; strict refuses, non-strict takes B
    const mixed_args = [_]aot.Arg{ aot.ptr("X", "*fp32", x.ptr), aot.int("n", 63), aot.float("s", 0), aot.ptr("OUT", "*u64", e.out.ptr) };
    const strict = set.run(e.s, "_echo", .{ 1, 1, 1 }, &mixed_args, &consts);
    set.strict = false;
    try set.run(e.s, "_echo", .{ 1, 1, 1 }, &mixed_args, &consts);
    try e.s.synchronize();
    try e.out.download(0, std.mem.sliceAsBytes(&o));
    const fallback_tag = o[0];
    set.strict = true;
    const no_consts = set.run(e.s, "_echo", .{ 1, 1, 1 }, &mixed_args, &.{aot.ci("BLOCK", 64)});
    r.note("2 kit variants of _echo: aligned launch -> A (block 2 waves = 64), misaligned -> B (block 32), args and null hidden scratch as sent; no exact variant: strict -> {s}, non-strict fallback -> variant {x} ({d} fallback counted); unknown constexpr -> {s}", .{ if (strict) |_| "ran?!" else |y| @errorName(y), fallback_tag, set.fallbacks.load(.monotonic), if (no_consts) |_| "ran?!" else |y| @errorName(y) });
    try r.expect(errIs(strict, error.MissingTritonVariant) and fallback_tag == 0xB and set.fallbacks.load(.monotonic) == 1 and errIs(no_consts, error.MissingTritonVariant), "variant rules", .{});
}

fn strictLaunches(r: *Receipt, e: *const Env) !void {
    const k = try EchoParams.bind(e.kit, &.{});
    var a: gpu.Args = .{};
    a.add(@as(u8, 1));
    a.add(@as(u64, 2)); // the rest missing
    const short = gpu.launch.launch(k.f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, e.s, &a);
    const km = try EchoMixed.bind(e.kit, &.{});
    var m: abi_h.tf_ds_abi_mixed = std.mem.zeroes(abi_h.tf_ds_abi_mixed);
    m.u8 = 1;
    const wrong_block = km.launch(e.s, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, .{ .a = m, .out = e.out.ptr });
    const i = e.kit.find("abi.echo_mixed", &.{}).?;
    r.note("strict {}: a launch packing 2 of 8 parameters -> {s}; echo_mixed (block declared 64) at 128 -> {s}; echo_mixed launches counted {d}", .{ e.kit.strict, if (short) |_| "launched?!" else |x| @errorName(x), if (wrong_block) |_| "launched?!" else |x| @errorName(x), e.kit.launches(i) });
    try r.expect(e.kit.strict and errIs(short, error.Invalid) and errIs(wrong_block, error.Invalid) and e.kit.launches(i) >= 1, "strict checks", .{});
}

/// A copy of the kit with `edit` applied, opened as a second kit: it must be refused with `want`.
fn refusedCopy(r: *Receipt, e: *const Env, name: []const u8, want: anyerror, edit: *const fn (*const Env, []const u8) anyerror!void, features: gpu.kit.Features) !void {
    const dir = try std.fs.path.join(e.gpa, &.{ e.scratch, name });
    try copyKit(e, dir);
    try edit(e, dir);
    const got = gpu.Kit.open(e.gpa, e.io, e.d, dir, e.arch, features);
    if (got) |k| k.deinit() else |_| {}
    r.note("{s} -> {s}; ", .{ name, if (got) |_| "opened?!" else |x| @errorName(x) });
    try r.expect(errIs(got, want), "{s} was not refused with {s}", .{ name, @errorName(want) });
}

fn copyKit(e: *const Env, dir: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(e.io, try std.fs.path.join(e.gpa, &.{ dir, "objects" }));
    const text = try cwd.readFileAlloc(e.io, try std.fs.path.join(e.gpa, &.{ e.kit_dir, "manifest.json" }), e.gpa, .limited(64 << 20));
    try cwd.writeFile(e.io, .{ .sub_path = try std.fs.path.join(e.gpa, &.{ dir, "manifest.json" }), .data = text });
    for (e.kit.m.objects) |o| {
        const bytes = try cwd.readFileAlloc(e.io, try std.fs.path.join(e.gpa, &.{ e.kit_dir, o.file }), e.gpa, .limited(1 << 30));
        try cwd.writeFile(e.io, .{ .sub_path = try std.fs.path.join(e.gpa, &.{ dir, o.file }), .data = bytes });
    }
}

fn editManifest(e: *const Env, dir: []const u8, from: []const u8, to: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    const path = try std.fs.path.join(e.gpa, &.{ dir, "manifest.json" });
    const text = try cwd.readFileAlloc(e.io, path, e.gpa, .limited(64 << 20));
    if (std.mem.indexOf(u8, text, from) == null) return error.EditNotApplied;
    try cwd.writeFile(e.io, .{ .sub_path = path, .data = try std.mem.replaceOwned(u8, e.gpa, text, from, to) });
}

const Edits = struct {
    fn flipByte(e: *const Env, dir: []const u8) anyerror!void {
        const cwd = std.Io.Dir.cwd();
        const path = try std.fs.path.join(e.gpa, &.{ dir, e.kit.m.objects[0].file });
        const bytes = try cwd.readFileAlloc(e.io, path, e.gpa, .limited(1 << 30));
        bytes[bytes.len / 2] ^= 0x01;
        try cwd.writeFile(e.io, .{ .sub_path = path, .data = bytes });
    }
    fn nextAbi(e: *const Env, dir: []const u8) anyerror!void {
        var from: [32]u8 = undefined;
        var to: [32]u8 = undefined;
        try editManifest(e, dir, try std.fmt.bufPrint(&from, "\"abi_version\": {d}", .{gpu.kit.abi_version}), try std.fmt.bufPrint(&to, "\"abi_version\": {d}", .{gpu.kit.abi_version + 1}));
    }
    fn requireAbsent(e: *const Env, dir: []const u8) anyerror!void {
        try editManifest(e, dir, "\"drafts\": []", "\"drafts\": [\"abi.absent\"]");
    }
    /// The first kernel's kernarg_size, 8 bytes more than its object says.
    fn lieAboutArgs(e: *const Env, dir: []const u8) anyerror!void {
        const key = "\"kernarg_size\": ";
        const k0 = e.kit.m.kernels[0].kernarg_size;
        var from: [48]u8 = undefined;
        var to: [48]u8 = undefined;
        try editManifest(e, dir, try std.fmt.bufPrint(&from, key ++ "{d},", .{k0}), try std.fmt.bufPrint(&to, key ++ "{d},", .{k0 + 8}));
    }
};

fn refusals(r: *Receipt, e: *const Env) !void {
    try refusedCopy(r, e, "tampered-object", error.HashMismatch, Edits.flipByte, .{});
    try refusedCopy(r, e, "next-abi-version", error.AbiMismatch, Edits.nextAbi, .{});
    try refusedCopy(r, e, "missing-required-kernel", error.MissingKernel, Edits.requireAbsent, .{ .drafts = true });
    try refusedCopy(r, e, "manifest-disagrees-with-metadata", error.MetadataMismatch, Edits.lieAboutArgs, .{});
    const other_arch = gpu.Kit.open(e.gpa, e.io, e.d, e.kit_dir, "gfx1150", .{});
    if (other_arch) |k| k.deinit() else |_| {}
    r.note("same kit for gfx1150 -> {s}", .{if (other_arch) |_| "opened?!" else |x| @errorName(x)});
    try r.expect(errIs(other_arch, error.ArchMismatch), "arch", .{});
}

const Extra = struct { e: *const Env, dir: []const u8 };

/// Opens `dir` as a second kit (after the first is closed for strict checks), builds its Triton set, reports counts.
fn extraKit(r: *Receipt, x: *const Extra) !void {
    const e = x.e;
    const dir = x.dir;
    const k = try gpu.Kit.open(e.gpa, e.io, e.d, dir, e.arch, .{});
    defer k.deinit();
    var set = try gpu.aot.Set.fromKit(e.gpa, k);
    defer set.deinit();
    var fns: std.StringHashMapUnmanaged(void) = .empty;
    var spills: usize = 0;
    for (k.entries) |entry| {
        if (entry.k.triton) |t| try fns.put(e.gpa, t.@"fn", {});
        if (entry.k.scratch > 0) spills += 1;
    }
    r.note("{s}: {d} objects loaded (SHA-256, ELF gfx1151, metadata checked), {d} kernels resolved, {d} Triton variants of {d} functions in the set, {d} spill registers to scratch, manifest sha256 {s}", .{ dir, k.m.objects.len, k.entries.len, set.variants.len, fns.count(), spills, &k.manifest_sha256 });
    try r.expect(set.variants.len == k.entries.len and set.variants.len > 0, "not every entry is a Triton variant", .{});
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.arena.allocator(); // everything here lives until exit
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    var kit_dir: ?[]const u8 = null;
    var scratch: ?[]const u8 = null;
    var receipt_path: ?[]const u8 = null;
    var extra_kit: ?[]const u8 = null;
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--kit")) kit_dir = args[i + 1];
        if (std.mem.eql(u8, args[i], "--scratch")) scratch = args[i + 1];
        if (std.mem.eql(u8, args[i], "--receipt")) receipt_path = args[i + 1];
        if (std.mem.eql(u8, args[i], "--extra-kit")) extra_kit = args[i + 1];
    }
    if (kit_dir == null or scratch == null) {
        std.debug.print("usage: tf-dsv41-hip-abi-test --kit DIR --scratch DIR [--receipt PATH]\n", .{});
        return 2;
    }
    var d = try gpu.Driver.open();
    defer d.close();
    var ctx = try gpu.Context.init(&d, 0);
    defer ctx.deinit();
    const caps = try ctx.caps();
    const kit = try gpu.Kit.open(gpa, io, &d, kit_dir.?, caps.archName(), .{});
    var kit_open = true;
    defer if (kit_open) kit.deinit();
    var s = try gpu.Stream.init(&d, true);
    defer s.deinit();
    var out = try gpu.DeviceBuffer.alloc(&d, 4096);
    defer out.free();
    const env: Env = .{ .gpa = gpa, .io = io, .d = &d, .arch = caps.archName(), .kit = kit, .kit_dir = kit_dir.?, .scratch = scratch.?, .s = s, .out = out };
    std.debug.print("kit {s}: {d} objects, {d} kernels, ABI {d}, manifest sha256 {s}, strict {}\n", .{ kit_dir.?, kit.m.objects.len, kit.m.kernels.len, kit.m.abi_version, &kit.manifest_sha256, kit.strict });
    var r: Receipt = .{ .gpa = gpa };
    try r.run("by_value_structs", structs, &env);
    try r.run("exl3_structs", exl3Structs, &env);
    try r.run("parameter_lists", params, &env);
    try r.run("triton_variant_set", triton, &env);
    try r.run("strict_launch_checks", strictLaunches, &env);
    try r.run("kit_refusals", refusals, &env);
    const manifest_sha = kit.manifest_sha256;
    const kit_counts = .{ kit.m.objects.len, kit.m.kernels.len, kit.m.abi_version, kit.strict };
    if (extra_kit) |dir| {
        // one kit a process holds the strict hook: close the test kit first
        kit.deinit();
        kit_open = false;
        const x: Extra = .{ .e = &env, .dir = dir };
        try r.run("extra_kit_loads", extraKit, &x);
    }
    std.debug.print("{d} passed, {d} failed\n", .{ r.passed, r.failed });
    if (receipt_path) |p| {
        const text = try std.fmt.allocPrint(gpa, "{{\n  \"receipt\": \"tf-dsv41-hip-abi-test\",\n  \"schema\": 1,\n  \"arch\": \"{s}\",\n  \"abi_version\": {d},\n  \"kit\": \"{s}\",\n  \"kit_manifest_sha256\": \"{s}\",\n  \"kit_objects\": {d},\n  \"kit_kernels\": {d},\n  \"strict\": {},\n  \"extra_kit\": \"{s}\",\n  \"passed\": {d},\n  \"failed\": {d},\n  \"checks\": [\n{s}\n  ]\n}}\n", .{ caps.archName(), kit_counts[2], kit_dir.?, &manifest_sha, kit_counts[0], kit_counts[1], kit_counts[3], extra_kit orelse "", r.passed, r.failed, r.checks.items });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = text });
    }
    return if (r.failed == 0) 0 else 1;
}
