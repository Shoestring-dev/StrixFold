//! tf-dsv41-hip-smoke [--receipt PATH]: the HIP runtime behind the dsv41 family on the real GPU. Every check compares
//! against values computed on the host; the receipt JSON lists each check with pass/fail and what it saw. Run it under
//! the GPU lock (tools/dsv41-hip/run_gpu_tests.sh). Exit code 0 only when every check passes.

const std = @import("std");
const gpu = @import("gpu");
const kit = @import("smoke_kernels");

const Fail = error{CheckFailed};

/// Whether `x` is the error `want` (an error union compared without unwrapping it).
fn errIs(x: anytype, want: anyerror) bool {
    if (x) |_| return false else |e| return e == want;
}

const Receipt = struct {
    gpa: std.mem.Allocator,
    checks: std.ArrayList(u8) = .empty,
    passed: u32 = 0,
    failed: u32 = 0,
    detail: [512]u8 = undefined,
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
    d: *const gpu.Driver,
    ctx: *const gpu.Context,
    m: gpu.Module,
    s: gpu.Stream,
    s2: gpu.Stream,
    clock_khz: u64,

    fn f(e: *const Env, name: [:0]const u8) !gpu.Function {
        return e.m.function(name);
    }

    fn ticks(e: *const Env, ms: u64) u64 {
        return ms * e.clock_khz;
    }
};

fn launch1(f: gpu.Function, s: gpu.Stream, grid: u32, block: u32, args: anytype) !void {
    var a: gpu.Args = .{};
    inline for (args) |v| a.add(v);
    try gpu.launch.launch(f, .{ .grid = .{ .x = grid }, .block = .{ .x = block } }, s, &a);
}

fn deviceCaps(r: *Receipt, e: *const Env) !void {
    const c = try e.ctx.caps();
    var full: [256]u8 = undefined;
    var nb: [256]u8 = undefined;
    const mi = try e.ctx.memInfo();
    r.note("arch {s} (runtime {s}) name {s} wave {d} cus {d} lds/block {d} integrated {} wall clock {d} Hz hip {d} mem free {d} total {d} header-checked prototypes {d}", .{
        c.archName(), try e.ctx.arch(&full), try e.ctx.name(&nb), c.isa.wave, c.cus, c.lds_per_block, c.integrated, c.wall_clock_hz, try e.d.version(), mi.free, mi.total, gpu.abiChecked(),
    });
    try r.expect(std.mem.eql(u8, c.archName(), "gfx1151"), "arch is not gfx1151", .{});
    try r.expect(c.isa.wave == 32 and c.isa.wmma_f16 and c.cus > 0 and c.lds_per_block == 65536 and c.wall_clock_hz > 0, "caps", .{});
    try r.expect(gpu.abiChecked() == 63, "abi check object", .{});
}

fn imageAndSymbols(r: *Receipt, e: *const Env) !void {
    try gpu.checkImage(&kit.image, kit.arch);
    for ([_][:0]const u8{ "tf_smoke_probe", "tf_smoke_step", "tf_smoke_wait_write", "tf_smoke_wave", "tf_smoke_xor", "tf_smoke_read_table", "tf_smoke_f16x2_lanes" }) |n| _ = try e.f(n);
    const g = try e.m.global("tf_smoke_table");
    r.note("raw ELF gfx1151 {d} bytes, 7 kernels found, global {d} bytes", .{ kit.image.len, g.len });
    try r.expect(g.len == 16, "global size {d}", .{g.len});
    const absent = e.m.function("tf_smoke_absent");
    try r.expect(errIs(absent, error.NotFound) or errIs(absent, error.NoKernelImage), "absent symbol was found", .{});
}

fn probeArgs(r: *Receipt, e: *const Env) !void {
    const n: u32 = 1000; // not a multiple of the block: the tail is exercised
    var b = try gpu.DeviceBuffer.alloc(e.d, n * 4);
    defer b.free();
    try b.fill8(0xff, null);
    try launch1(try e.f("tf_smoke_probe"), e.s, (n + 255) / 256, 256, .{ b.ptr, n, @as(u32, 7), @as(u64, 5) << 32 });
    try e.s.synchronize();
    var out: [1000]u32 = undefined;
    try b.download(0, std.mem.sliceAsBytes(&out));
    for (out, 0..) |v, i| try r.expect(v == i + 12, "out[{d}] = {d}", .{ i, v });
    var w = try gpu.DeviceBuffer.alloc(e.d, 4);
    defer w.free();
    try launch1(try e.f("tf_smoke_wave"), e.s, 1, 64, .{w.ptr});
    try e.s.synchronize();
    var wave: [1]i32 = undefined;
    try w.download(0, std.mem.sliceAsBytes(&wave));
    r.note("1000 values from u32/u32/u64 args correct; warpSize {d}; launches so far {d}", .{ wave[0], gpu.launch.count });
    try r.expect(wave[0] == 32, "warpSize {d}", .{wave[0]});
}

fn copiesAndFills(r: *Receipt, e: *const Env) !void {
    const n = 4096;
    var a = try gpu.DeviceBuffer.alloc(e.d, n);
    defer a.free();
    var b = try gpu.DeviceBuffer.alloc(e.d, n);
    defer b.free();
    var host: [n]u8 = undefined;
    try a.fill8(0xab, null);
    try a.download(0, &host);
    for (host) |v| try r.expect(v == 0xab, "fill8 sync", .{});
    try a.fill32(0x01020304, e.s.handle);
    try e.s.synchronize();
    try a.download(0, &host);
    for (std.mem.bytesAsSlice(u32, &host)) |v| try r.expect(v == 0x01020304, "fill32 async {x}", .{v});
    for (&host, 0..) |*v, i| v.* = @truncate(i *% 31 +% 7);
    try a.upload(0, &host);
    try b.copyFrom(0, a.ptr, n, null);
    var back: [n]u8 = undefined;
    try b.download(0, &back);
    try r.expect(std.mem.eql(u8, &host, &back), "upload + D2D sync", .{});
    try gpu.fill8Async(e.d, e.s.handle, b.ptr + 100, 0x11, 50);
    try gpu.fill32Async(e.d, e.s.handle, b.ptr + 1024, 0xdeadbeef, 8);
    try gpu.copyD2DAsync(e.d, e.s.handle, b.ptr + 2048, a.ptr + 7, 300);
    try e.s.synchronize();
    try gpu.copyD2H(e.d, &back, b.ptr);
    for (back[100..150]) |v| try r.expect(v == 0x11, "fill8Async range", .{});
    try r.expect(std.mem.eql(u8, back[150..1024], host[150..1024]), "fill8Async overran", .{});
    for (std.mem.bytesAsSlice(u32, back[1024..1056])) |v| try r.expect(v == 0xdeadbeef, "fill32Async", .{});
    try r.expect(std.mem.eql(u8, back[2048..2348], host[7..307]), "copyD2DAsync", .{});
    var pin = try gpu.HostBuffer.alloc(e.d, n);
    defer pin.free();
    @memcpy(pin.bytes, &host);
    try b.uploadAsync(0, pin.bytes, e.s.handle);
    @memset(&back, 0);
    try e.s.synchronize();
    @memset(pin.bytes, 0);
    try b.downloadAsync(0, pin.bytes, e.s.handle);
    try e.s.synchronize();
    try r.expect(std.mem.eql(u8, pin.bytes, &host), "pinned async round trip", .{});
    try gpu.copyH2D(e.d, a.ptr + 4000, host[0..96]);
    try a.download(4000, back[0..96]);
    try r.expect(std.mem.eql(u8, back[0..96], host[0..96]), "copyH2D at offset", .{});
    try r.expect(errIs(a.at(n + 1), error.Invalid) and errIs(a.download(n - 2, back[0..4]), error.Invalid), "range checks", .{});
    r.note("sync/async fills, H2D/D2H/D2D sync and async, pinned round trip, offsets and range refusals", .{});
}

fn moduleGlobal(r: *Receipt, e: *const Env) !void {
    const g = try e.m.global("tf_smoke_table");
    const vals = [4]i32{ 1, 2, 3, -4 };
    try gpu.copyH2D(e.d, g.ptr, std.mem.sliceAsBytes(&vals));
    var o = try gpu.DeviceBuffer.alloc(e.d, 16);
    defer o.free();
    try launch1(try e.f("tf_smoke_read_table"), e.s, 1, 32, .{o.ptr});
    try e.s.synchronize();
    var out: [4]i32 = undefined;
    try o.download(0, std.mem.sliceAsBytes(&out));
    r.note("global written by the host, read by a kernel: {any}", .{out});
    try r.expect(std.mem.eql(i32, &out, &.{ 2, 4, 6, -8 }), "values", .{});
}

fn streamsAndEvents(r: *Receipt, e: *const Env) !void {
    var buf = try gpu.DeviceBuffer.alloc(e.d, 8);
    defer buf.free();
    try buf.fill8(0, null);
    var ev = try gpu.Event.init(e.d, false);
    defer ev.deinit();
    try launch1(try e.f("tf_smoke_wait_write"), e.s, 1, 32, .{ buf.ptr, @as(u32, 77), e.ticks(30) });
    try ev.record(e.s);
    const busy_ev = !try ev.done();
    const busy_s = !try e.s.done();
    try e.s2.wait(ev);
    try launch1(try e.f("tf_smoke_xor"), e.s2, 1, 32, .{ buf.ptr, buf.ptr + 4, @as(u32, 1) });
    try e.s2.synchronize();
    var out: [2]u32 = undefined;
    try buf.download(0, std.mem.sliceAsBytes(&out));
    r.note("event pending after launch {} stream pending {}; consumer on stream 2 saw {d} after waiting on the event", .{ busy_ev, busy_s, out[0] });
    try r.expect(busy_ev and busy_s, "queued work was not visible as pending", .{});
    try r.expect(out[0] == 77 and out[1] == 77 ^ 0x5a5a5a5a, "cross-stream order {d} {d}", .{ out[0], out[1] });
    try r.expect(try ev.done() and try e.s.done(), "done after completion", .{});
    var t0 = try gpu.Event.init(e.d, true);
    defer t0.deinit();
    var t1 = try gpu.Event.init(e.d, true);
    defer t1.deinit();
    try t0.record(e.s);
    try launch1(try e.f("tf_smoke_wait_write"), e.s, 1, 32, .{ buf.ptr, @as(u32, 1), e.ticks(5) });
    try t1.record(e.s);
    try t1.synchronize();
    const ms = try gpu.Event.elapsedMs(t0, t1);
    r.note("; timed 5 ms of wall clock as {d:.3} ms", .{ms});
    try r.expect(ms >= 4.5 and ms < 50, "elapsed {d}", .{ms});
}

fn boundedWait(r: *Receipt, e: *const Env) !void {
    var buf = try gpu.DeviceBuffer.alloc(e.d, 4);
    defer buf.free();
    try buf.fill8(0, null);
    try launch1(try e.f("tf_smoke_wait_write"), e.s, 1, 32, .{ buf.ptr, @as(u32, 9), e.ticks(200) });
    const timed_out = errIs(gpu.synchronizeFor(e.s, 2 * std.time.ns_per_ms), error.Timeout);
    try e.s.synchronize();
    var v: [1]u32 = undefined;
    try buf.download(0, std.mem.sliceAsBytes(&v));
    r.note("2 ms bound on 200 ms of work timed out {}; work then completed with {d}", .{ timed_out, v[0] });
    try r.expect(timed_out and v[0] == 9, "bounded wait", .{});
    try gpu.synchronizeFor(e.s, std.time.ns_per_s);
}

fn staging(r: *Receipt, e: *const Env) !void {
    var st = try gpu.Staging.init(e.d, 64 << 10);
    defer st.deinit();
    var a = try gpu.DeviceBuffer.alloc(e.d, 1024);
    defer a.free();
    var b = try gpu.DeviceBuffer.alloc(e.d, 1024);
    defer b.free();
    var flag = try gpu.DeviceBuffer.alloc(e.d, 4);
    defer flag.free();
    // the stream is busy for 50 ms, so neither copy can have run when the caller's array is overwritten
    try launch1(try e.f("tf_smoke_wait_write"), e.s, 1, 32, .{ flag.ptr, @as(u32, 1), e.ticks(50) });
    var src: [256]u32 = undefined;
    for (&src, 0..) |*v, i| v.* = @intCast(i * 3 + 1);
    try st.upload(e.s, a.ptr, std.mem.sliceAsBytes(&src));
    for (&src, 0..) |*v, i| v.* = @intCast(0xa0000000 + i);
    try st.upload(e.s, b.ptr, std.mem.sliceAsBytes(&src));
    for (&src) |*v| v.* = 0xdeaddead; // what a non-staged copy would carry
    const pending = !try e.s.done();
    try e.s.synchronize();
    var got: [256]u32 = undefined;
    try a.download(0, std.mem.sliceAsBytes(&got));
    for (got, 0..) |v, i| try r.expect(v == i * 3 + 1, "first staged upload [{d}] = {x}", .{ i, v });
    try b.download(0, std.mem.sliceAsBytes(&got));
    for (got, 0..) |v, i| try r.expect(v == 0xa0000000 + i, "second staged upload [{d}] = {x}", .{ i, v });
    // many small uploads through a 64 KiB ring (wrap and waits), then one larger than the ring (pieces)
    const big = 300 << 10;
    var c = try gpu.DeviceBuffer.alloc(e.d, big);
    defer c.free();
    var piece: [1000]u8 = undefined;
    for (0..200) |k| {
        for (&piece, 0..) |*v, i| v.* = @truncate(k *% 7 +% i);
        try st.upload(e.s, c.ptr + k * 1000, &piece);
    }
    try e.s.synchronize();
    const host = try r.gpa.alloc(u8, big);
    defer r.gpa.free(host);
    try c.download(0, host);
    for (0..200) |k| for (0..1000) |i| try r.expect(host[k * 1000 + i] == @as(u8, @truncate(k *% 7 +% i)), "small upload {d}", .{k});
    for (host, 0..) |*v, i| v.* = @truncate(i *% 13);
    try st.upload(e.s, c.ptr, host);
    @memset(host, 0);
    try e.s.synchronize();
    try c.download(0, host);
    for (host, 0..) |v, i| try r.expect(v == @as(u8, @truncate(i *% 13)), "large upload at {d}", .{i});
    try gpu.graph.beginCapture(e.s, .thread_local);
    const refused = errIs(st.upload(e.s, a.ptr, std.mem.sliceAsBytes(&src)), error.InCapture);
    var gr = try gpu.graph.endCapture(e.s);
    gr.deinit();
    r.note("caller bytes overwritten while the stream was busy ({}): device holds the staged bytes; 200 x 1000 B and 300 KiB through a 64 KiB ring ({d} waits, {d} bytes staged); upload during capture refused {}", .{ pending, st.waits, st.staged, refused });
    try r.expect(pending and refused and st.waits > 0, "staging conditions", .{});
}

fn graphs(r: *Receipt, e: *const Env) !void {
    var counter = try gpu.DeviceBuffer.alloc(e.d, 8);
    defer counter.free();
    try counter.fill8(0, null);
    const step = try e.f("tf_smoke_step");
    const record = struct {
        fn go(s: gpu.Stream, f: gpu.Function, ptr: u64, adds: [4]u64) !gpu.graph.Graph {
            try gpu.graph.beginCapture(s, .thread_local);
            for (adds) |x| try launch1(f, s, 1, 32, .{ ptr, x });
            return gpu.graph.endCapture(s);
        }
    };
    var g1 = try record.go(e.s, step, counter.ptr, .{ 1, 2, 3, 4 });
    defer g1.deinit();
    const nodes = try g1.nodeCount();
    var ex = try g1.instantiate();
    defer ex.deinit();
    try ex.upload(e.s);
    for (0..3) |_| try ex.launchOn(e.s);
    try e.s.synchronize();
    var v: [1]u64 = undefined;
    try counter.download(0, std.mem.sliceAsBytes(&v));
    const after_replay = v[0];
    var g2 = try record.go(e.s, step, counter.ptr, .{ 10, 20, 30, 40 });
    defer g2.deinit();
    const verdict = try ex.update(g2);
    try ex.launchOn(e.s);
    try e.s.synchronize();
    try counter.download(0, std.mem.sliceAsBytes(&v));
    const after_update = v[0];
    // node handles are the instantiated graph's (g1's), whatever the exec was later updated from
    var nb: [8]gpu.graph.Node = undefined;
    const ns = try g1.nodes(&nb);
    var a: gpu.Args = .{};
    a.add(counter.ptr);
    a.add(@as(u64, 1000));
    try ex.setKernel(ns[0], step, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, &a);
    try ex.launchOn(e.s);
    try e.s.synchronize();
    try counter.download(0, std.mem.sliceAsBytes(&v));
    r.note("captured {d} nodes; 3 replays -> {d}; exec update {s} -> {d}; node set -> {d}", .{ nodes, after_replay, @tagName(verdict), after_update, v[0] });
    try r.expect(nodes == 4 and after_replay == 30 and verdict == .success and after_update == 130 and v[0] == 1220, "graph values", .{});
}

fn mappedHost(r: *Receipt, e: *const Env) !void {
    const n = 4096;
    var out = try gpu.DeviceBuffer.alloc(e.d, n * 4);
    defer out.free();
    const xor = try e.f("tf_smoke_xor");
    for ([_]gpu.HostFlags{ .{ .mapped = true, .coherent = true }, .{ .mapped = true } }) |flags| {
        var h = try gpu.HostBuffer.allocFlags(e.d, n * 4, flags);
        defer h.free();
        const words = h.slice(u32);
        for (words, 0..) |*w, i| w.* = @intCast(i * 2654435761 % 4294967291);
        const dev = try h.device();
        try launch1(xor, e.s, n / 256, 256, .{ dev, out.ptr, @as(u32, n) });
        try launch1(xor, e.s, n / 256, 256, .{ out.ptr, dev, @as(u32, n) });
        try e.s.synchronize();
        var host_out: [n]u32 = undefined;
        try out.download(0, std.mem.sliceAsBytes(&host_out));
        for (words, host_out, 0..) |w, o, i| {
            const want: u32 = @intCast(i * 2654435761 % 4294967291);
            try r.expect(o == want ^ 0x5a5a5a5a and w == want, "mapped {s} word {d}", .{ if (flags.coherent) "coherent" else "non-coherent", i });
        }
    }
    // registered pages (a transport ring's mmap)
    const pages = try std.posix.mmap(null, 64 << 10, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(pages);
    var reg = try gpu.HostBuffer.register(e.d, pages);
    defer reg.free();
    const rw = reg.slice(u32);
    for (rw, 0..) |*w, i| w.* = @intCast(i);
    try launch1(xor, e.s, @intCast(rw.len / 256), 256, .{ try reg.device(), out.ptr, @as(u32, @intCast(rw.len)) });
    try e.s.synchronize();
    var first: [16]u32 = undefined;
    try out.download(0, std.mem.sliceAsBytes(&first));
    for (first, 0..) |o, i| try r.expect(o == i ^ 0x5a5a5a5a, "registered word {d}", .{i});
    r.note("coherent and non-coherent mapped host memory read and written by kernels at kernel boundaries; registered mmap pages read by a kernel (concurrent visibility is w-link's test)", .{});
}

fn allocators(r: *Receipt, e: *const Env) !void {
    for ([_]gpu.DeviceKind{ .default, .finegrained, .uncached }) |k| {
        var b = try gpu.DeviceBuffer.allocKind(e.d, 1 << 20, k);
        defer b.free();
        try b.fill32(0x12345678, e.s.handle);
        try e.s.synchronize();
        var w: [4]u32 = undefined;
        try b.download((1 << 20) - 16, std.mem.sliceAsBytes(&w));
        for (w) |x| try r.expect(x == 0x12345678, "{s} buffer", .{@tagName(k)});
    }
    const before = gpu.usage(false);
    const huge = gpu.DeviceBuffer.alloc(e.d, 1 << 50);
    r.note("hipMalloc, fine-grained and uncached 1 MiB buffers filled and read; 1 PiB request -> {s}; usage counters unchanged {}", .{ if (huge) |_| "allocated?!" else |x| @errorName(x), gpu.usage(false).device == before.device });
    if (huge) |h| {
        var hh = h;
        hh.free();
    } else |_| {}
    try r.expect(errIs(huge, error.OutOfDeviceMemory), "out-of-memory mapping", .{});
}

fn refusals(r: *Receipt, e: *const Env) !void {
    var a: gpu.Args = .{};
    const pdl = gpu.launch.launch(try e.f("tf_smoke_wave"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 }, .pdl = true }, e.s, &a);
    var other: [kit.image.len]u8 align(8) = kit.image;
    std.mem.writeInt(u32, other[48..52], (std.mem.readInt(u32, other[48..52], .little) & ~@as(u32, 0xff)) | 0x43, .little);
    const checked = gpu.checkImage(&other, "gfx1151");
    const loaded = gpu.Module.load(e.d, &other);
    if (loaded) |m| {
        var mm = m;
        mm.unload();
    } else |_| {}
    r.note("PDL launch -> {s}; image patched to another gfx: checkImage -> {s}, runtime load -> {s}", .{ if (pdl) |_| "launched?!" else |x| @errorName(x), if (checked) |_| "accepted?!" else |x| @errorName(x), if (loaded) |_| "loaded" else |x| @errorName(x) });
    try r.expect(errIs(pdl, error.Unsupported) and errIs(checked, error.NoKernelImage), "refusals", .{});
}

/// clang 23 / HIP 7.15's f16x2 lane bit_cast bug: recorded as present or absent; the check passes when the memcpy idiom
/// reads both lanes right (tf-hip-kit lint refuses unreviewed bit_cast uses in kernel sources).
fn f16x2Lanes(r: *Receipt, e: *const Env) !void {
    const in = [2]u16{ 0x3c00, 0xc000 }; // 1.0, -2.0
    var b = try gpu.DeviceBuffer.alloc(e.d, 32);
    defer b.free();
    try b.upload(0, std.mem.sliceAsBytes(&in));
    try launch1(try e.f("tf_smoke_f16x2_lanes"), e.s, 1, 32, .{ b.ptr, b.ptr + 16 });
    try e.s.synchronize();
    var o: [4]u32 = undefined;
    try b.download(16, std.mem.sliceAsBytes(&o));
    const bug = o[0] != 0xc000;
    r.note("__builtin_bit_cast(uint16_t, f16x2[1]) -> {x} (want c000): lane bug {s}; lane 0 -> {x}; memcpy lanes 1, 0 -> {x}, {x}", .{ o[0], if (bug) "PRESENT on this toolchain" else "absent", o[1], o[2], o[3] });
    try r.expect(o[2] == 0xc000 and o[3] == 0x3c00, "the memcpy idiom read the wrong lanes", .{});
}

fn healthy(r: *Receipt, _: *const Env) !void {
    r.note("device fault state: {?d}", .{gpu.health()});
    try r.expect(gpu.health() == null, "a device fault was recorded", .{});
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.arena.allocator(); // the receipt's strings live until exit
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var receipt_path: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--receipt") and i + 1 < args.len) {
            receipt_path = args[i + 1];
            i += 1;
        }
    }
    var d = try gpu.Driver.open();
    defer d.close();
    var ctx = try gpu.Context.init(&d, 0);
    defer ctx.deinit();
    var m = try gpu.Module.load(&d, &kit.image);
    defer m.unload();
    var s = try gpu.Stream.init(&d, true);
    defer s.deinit();
    var s2 = try gpu.Stream.init(&d, true);
    defer s2.deinit();
    const env: Env = .{ .d = &d, .ctx = &ctx, .m = m, .s = s, .s2 = s2, .clock_khz = @intCast(try ctx.attribute(.wall_clock_rate)) };
    var r: Receipt = .{ .gpa = gpa };
    try r.run("device_caps", deviceCaps, &env);
    try r.run("code_object_and_symbols", imageAndSymbols, &env);
    try r.run("launch_args_and_wave", probeArgs, &env);
    try r.run("copies_and_fills", copiesAndFills, &env);
    try r.run("module_global", moduleGlobal, &env);
    try r.run("streams_events_timing", streamsAndEvents, &env);
    try r.run("bounded_wait", boundedWait, &env);
    try r.run("pageable_staging", staging, &env);
    try r.run("graph_capture_replay_update", graphs, &env);
    try r.run("mapped_and_registered_host_memory", mappedHost, &env);
    try r.run("allocators_and_oom", allocators, &env);
    try r.run("cuda_only_refusals", refusals, &env);
    try r.run("toolchain_f16x2_lane_bitcast", f16x2Lanes, &env);
    try r.run("device_health", healthy, &env);
    std.debug.print("{d} passed, {d} failed\n", .{ r.passed, r.failed });
    if (receipt_path) |p| {
        const buf = try std.fmt.allocPrint(gpa, "{{\n  \"receipt\": \"tf-dsv41-hip-smoke\",\n  \"schema\": 1,\n  \"arch\": \"{s}\",\n  \"kernel_object_bytes\": {d},\n  \"passed\": {d},\n  \"failed\": {d},\n  \"checks\": [\n{s}\n  ]\n}}\n", .{ kit.arch, kit.image.len, r.passed, r.failed, r.checks.items });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = buf });
    }
    return if (r.failed == 0) 0 else 1;
}
