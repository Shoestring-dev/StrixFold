//! The TP ranks' collectives as the family calls them (tools/dsv41-hip/ADAPTER.md section 5; the interface agreed with
//! w-link on 2026-10-19). Every model reduction is an all-gather of fp32 partials that each rank then adds in rank order
//! (DESIGN.md section 3), so a collective here only moves bytes: a stream-ordered all-gather in rank order, the
//! four-node split's point-to-point exchange (CUDA's NCCL only), and abort. NCCL implements it on CUDA (comm.zig
//! Comm.coll); w-link's host-staged TCP transport on HIP, handed to model.open as Options.open_coll.
//!
//! The contract an implementation keeps:
//! - Every rank issues the same sequence of calls: the same site and byte count in the same order. A transport that
//!   carries a header (TCP) checks the site, its layer, the bytes and its own generation against the peer's and fails
//!   the call with Mismatch on any difference; NCCL carries none and does not check.
//! - The first failure poisons the Coll: that call and every later one return the same error.
//! - abort is idempotent and may be called from any thread; calls blocked in the transport return (Aborted, or the
//!   error that poisoned it first).
//! - close releases everything in a bounded time; a call still in flight is aborted first.
//! - Deadlines (progress-based) and generations are the implementation's own; they are not parameters.
const std = @import("std");
const gpu = @import("gpu");
const Fan = @import("fan.zig").Fan;

/// The element types the family moves (the names NCCL's DataType gives them); a collective only counts their bytes.
pub const DType = enum {
    i8,
    u8,
    i32,
    u32,
    i64,
    u64,
    f16,
    f32,
    f64,
    bf16,

    pub fn size(t: DType) usize {
        return switch (t) {
            .i8, .u8 => 1,
            .f16, .bf16 => 2,
            .i32, .u32, .f32 => 4,
            .i64, .u64, .f64 => 8,
        };
    }
};

/// A collective's failure, the same on every later call (the Coll is poisoned).
pub const Error = error{
    /// the transport failed (a socket or device copy error, NCCL's error code)
    CollectiveFailed,
    /// the peer closed or reset its connection
    PeerLost,
    /// no progress within the implementation's deadline
    Timeout,
    /// the peer issued a different call (site, layer, bytes or generation)
    Mismatch,
    /// abort was called
    Aborted,
};

/// Why a collective stopped: what a rank tells its peer and its client.
pub const Reason = enum { cancelled, peer_lost, timeout, device_fault, shutdown };

/// Where in the forward a collective is issued: every all-gather the family makes, by pass. The decode rounds' and the
/// drafter's are the gathers among round.Point / draft.Point; `layer` is the layer (the drafter's block, the Markov
/// step; the head's is the layer count).
pub const Phase = enum(u16) {
    round_engram = 1, // round.zig stretch: Point.engram_gather
    round_attn, // Point.attn_gather
    round_moe, // Point.moe_gather
    round_head, // Point.head_gather
    prompt_engram, // prompt.zig engramApply (a prompt chunk)
    prompt_attn, // prompt.gather after attention
    prompt_moe, // prompt.gather after the MoE
    prompt_head, // prompt.head
    draft_attn, // draft.zig body: draft.Point.attn_gather
    draft_moe, // draft.Point.moe_gather
    draft_markov, // the Markov steps' bests (layer: the step)
    ring_setup, // CUDA: the RDMA rings' queue-pair infos and their start barrier
    two_parts, // CUDA world 4: prompt2d's parts
    two_pair, // CUDA world 4: prompt2d's pair exchanges
    test_only = 0xffff,
};

pub const Site = struct {
    phase: Phase,
    layer: u16,

    pub fn at(phase: Phase, layer: usize) Site {
        return .{ .phase = phase, .layer = @intCast(layer) };
    }
};

pub const Coll = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    rank: u32,
    world: u32,
    /// Its all-gather may be captured into a graph (NCCL, or a device-proxy transport); a host-staged transport is not,
    /// so the rounds run eagerly with it (TF_DS_GRAPHS=1 is refused at open).
    capturable: bool,
    /// TF_DS_COLL_TIME's per-kind wall time (Timer; null, the default: the calls go straight to the transport).
    timer: ?*Timer = null,

    pub const VTable = struct {
        /// recv[r * bytes .. (r + 1) * bytes] = rank r's send, for every rank in rank order. Stream-ordered: reads `send`
        /// after the work already queued on `stream`; work queued after it sees `recv`. A host-staged transport may block
        /// the calling thread until `recv` is written.
        all_gather: *const fn (ptr: *anyopaque, stream: gpu.Stream, site: Site, send: u64, recv: u64, bytes: usize) Error!void,
        /// `send_bytes` to rank `peer` and `recv_bytes` from it, stream-ordered (the four-node split; null: none).
        exchange: ?*const fn (ptr: *anyopaque, stream: gpu.Stream, site: Site, send: u64, send_bytes: usize, recv: u64, recv_bytes: usize, peer: u32) Error!void = null,
        /// Fails every call in flight and after with Aborted (unless already poisoned); idempotent, any thread.
        abort: *const fn (ptr: *anyopaque, reason: Reason) void,
        /// Releases the transport in a bounded time, aborting a call in flight.
        close: *const fn (ptr: *anyopaque) void,
        /// The all-gather in two halves (Coll.gatherBegin / gatherEnd; null: the gather runs whole at its end).
        gather_begin: ?*const fn (ptr: *anyopaque, stream: gpu.Stream, site: Site, send: u64, recv: u64, bytes: usize) Error!void = null,
        gather_end: ?*const fn (ptr: *anyopaque, stream: gpu.Stream) Error!void = null,
        /// The transport's failure, if it has failed (null: it reports failures only from its calls). A transport whose
        /// gathers run without a host call (a device proxy, a captured graph) is checked after the stream synchronizes.
        check: ?*const fn (ptr: *anyopaque) Error!void = null,
        /// How many gathers this rank's transport has moved so far (diagnostics; null: it does not count).
        counts: ?*const fn (ptr: *anyopaque) Counts = null,
    };

    /// A transport's running totals, equal on both ranks whenever their streams stand at the same point: device-proxy
    /// gathers staged (tcp_proxy's slab sequence) and the exchanges of each of its wires (both proxy cables, tcp_host's).
    pub const Counts = struct { staged: u64 = 0, cable1: u64 = 0, cable2: u64 = 0, host: u64 = 0 };

    pub fn counts(c: Coll) Counts {
        return if (c.vtable.counts) |f| f(c.ptr) else .{};
    }

    /// After the stream synchronized: fails with the transport's error when a gather the host did not call (a captured
    /// one, a device wait that gave up at its deadline) failed, so no result of that work is used.
    pub fn check(c: Coll) Error!void {
        if (c.vtable.check) |f| return f(c.ptr);
    }

    /// `count` elements of `dtype` from every rank into `recv`, rank after rank, at `site`.
    pub fn allGather(c: Coll, site: Site, send: u64, recv: u64, count: usize, dtype: DType, stream: gpu.Stream) Error!void {
        const t = c.timer orelse return c.vtable.all_gather(c.ptr, stream, site, send, recv, count * dtype.size());
        const t0 = try t.start(stream);
        try c.vtable.all_gather(c.ptr, stream, site, send, recv, count * dtype.size());
        try t.stop(stream, .all_gather, t0);
    }

    /// A begun all-gather (gatherBegin), finished by gatherEnd.
    pub const Pending = struct { site: Site, send: u64, recv: u64, bytes: usize };

    /// allGather split so the bytes can move while the GPU runs other work: gatherBegin takes `send` as the work
    /// queued on `stream` so far leaves it and returns without waiting for it; work queued between the two calls must
    /// neither write `send` nor touch `recv`; gatherEnd completes the gather (a host-staged transport blocks there
    /// while the bytes move), and work queued on `stream` after it sees `recv`. One begun gather at a time per Coll and
    /// no other call on it in between; both ranks pair them alike. A transport without the split gathers at the end
    /// (the same bytes, no overlap).
    pub fn gatherBegin(c: Coll, site: Site, send: u64, recv: u64, count: usize, dtype: DType, stream: gpu.Stream) Error!Pending {
        const p: Pending = .{ .site = site, .send = send, .recv = recv, .bytes = count * dtype.size() };
        const t = c.timer orelse {
            if (c.vtable.gather_begin) |f| try f(c.ptr, stream, site, send, recv, p.bytes);
            return p;
        };
        const t0 = try t.start(stream);
        if (c.vtable.gather_begin) |f| try f(c.ptr, stream, site, send, recv, p.bytes);
        try t.stopBegin(stream, t0);
        return p;
    }

    pub fn gatherEnd(c: Coll, p: Pending, stream: gpu.Stream) Error!void {
        const t = c.timer orelse {
            if (c.vtable.gather_end) |f| return f(c.ptr, stream);
            return c.vtable.all_gather(c.ptr, stream, p.site, p.send, p.recv, p.bytes);
        };
        const t0 = try t.start(stream);
        if (c.vtable.gather_end) |f| try f(c.ptr, stream) else try c.vtable.all_gather(c.ptr, stream, p.site, p.send, p.recv, p.bytes);
        try t.stop(stream, .prompt_gather, t0);
    }

    /// `count` elements to rank `peer` and `rcount` from it (the four-node split's pairs).
    pub fn exchange(c: Coll, site: Site, send: u64, count: usize, recv: u64, rcount: usize, dtype: DType, peer: u32, stream: gpu.Stream) (Error || error{Unsupported})!void {
        const f = c.vtable.exchange orelse return error.Unsupported;
        const t = c.timer orelse return f(c.ptr, stream, site, send, count * dtype.size(), recv, rcount * dtype.size(), peer);
        const t0 = try t.start(stream);
        try f(c.ptr, stream, site, send, count * dtype.size(), recv, rcount * dtype.size(), peer);
        try t.stop(stream, .exchange, t0);
    }

    pub fn abort(c: Coll, reason: Reason) void {
        c.vtable.abort(c.ptr, reason);
    }

    pub fn close(c: Coll) void {
        c.vtable.close(c.ptr);
    }
};

/// TF_DS_COLL_TIME (off by default, rank 0 only): host wall time in each collective call, by kind, pass and rounds.
pub const Timer = struct {
    /// call: the time inside the call (tcp_host's wire, else the enqueue); sync: the stream drained around each call.
    pub const Mode = enum { call, sync };
    pub const Kind = enum { all_gather, exchange, prompt_gather };
    pub const kinds = 3;
    /// Calls and nanoseconds by Kind.
    pub const Sums = struct {
        calls: [kinds]u64 = @splat(0),
        ns: [kinds]u64 = @splat(0),
        pub fn totalNs(s: Sums) u64 {
            return s.ns[0] + s.ns[1] + s.ns[2];
        }
    };

    mode: Mode,
    /// The drain sync mode calls (the stream's synchronize; a host test's fake).
    drain: *const fn (stream: gpu.Stream) Error!void = drainStream,
    /// Rank 0's lanes sets it while a prompt pass slice runs: the calls go to `pass`, else to `rounds`.
    in_pass: bool = false,
    pass: Sums = .{},
    rounds: Sums = .{},
    /// Decode rounds since the last line (rank 0's lanes counts them).
    round_count: u64 = 0,
    /// A begun split gather's time so far (gatherBegin's call), added at its gatherEnd.
    begun_ns: u64 = 0,

    /// TF_DS_COLL_TIME's value: null (off) for unset, "" or "0"; "1" call; "sync" sync; anything else BadSetting.
    pub fn parseMode(v: ?[]const u8) error{BadSetting}!?Mode {
        const s = v orelse return null;
        if (s.len == 0 or std.mem.eql(u8, s, "0")) return null;
        if (std.mem.eql(u8, s, "1")) return .call;
        if (std.mem.eql(u8, s, "sync")) return .sync;
        return error.BadSetting;
    }

    fn drainStream(stream: gpu.Stream) Error!void {
        stream.synchronize() catch return error.CollectiveFailed;
    }

    pub fn nowNs() u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    fn start(t: *Timer, stream: gpu.Stream) Error!u64 {
        if (t.mode == .sync) try t.drain(stream);
        return nowNs();
    }

    fn stop(t: *Timer, stream: gpu.Stream, kind: Kind, t0: u64) Error!void {
        if (t.mode == .sync) try t.drain(stream);
        const s = if (t.in_pass) &t.pass else &t.rounds;
        s.calls[@intFromEnum(kind)] += 1;
        s.ns[@intFromEnum(kind)] += nowNs() -| t0 + t.begun_ns;
        t.begun_ns = 0;
    }

    fn stopBegin(t: *Timer, stream: gpu.Stream, t0: u64) Error!void {
        if (t.mode == .sync) try t.drain(stream);
        t.begun_ns = nowNs() -| t0;
    }

    /// The sums since the last line as rank 0's lanes line, then cleared (null with no timer: nothing is logged).
    pub fn line(timer: ?*Timer, buf: []u8, step: u64, rank: u32, at: []const u8) ?[]const u8 {
        const t = timer orelse return null;
        defer {
            t.pass = .{};
            t.rounds = .{};
            t.round_count = 0;
        }
        const p = t.pass;
        const r = t.rounds;
        const per = if (t.round_count == 0) 0 else ms(r.totalNs()) / @as(f64, @floatFromInt(t.round_count));
        return std.fmt.bufPrint(buf, "lanes: step {d} rank {d} coll time ({s}, {s}): pass {d:.3} ms (all_gather {d} {d:.3}, exchange {d} {d:.3}, prompt_gather {d} {d:.3}); rounds {d} {d:.3} ms (all_gather {d} {d:.3}, exchange {d} {d:.3}, prompt_gather {d} {d:.3}), {d:.3} ms a round", .{
            step,                     rank,       at,          if (t.mode == .sync) "sync" else "call",
            ms(p.totalNs()),          p.calls[0], ms(p.ns[0]), p.calls[1],
            ms(p.ns[1]),              p.calls[2], ms(p.ns[2]), t.round_count,
            ms(r.totalNs()),          r.calls[0], ms(r.ns[0]), r.calls[1],
            ms(r.ns[1]),              r.calls[2], ms(r.ns[2]), per,
        }) catch buf[0..0];
    }

    fn ms(ns: u64) f64 {
        return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
    }
};

/// What a rank brings to the peers' handshake (w-link's, inside Open): the transport compares it with every other rank's
/// and refuses the open on a difference, naming the field. `plan` is text (space-separated key=value): the settings that
/// decide which collectives a rank issues and with what sizes, so ranks that would issue different sequences never
/// start (model.zig Model.identity writes it).
pub const Identity = struct {
    rank: u32,
    world: u32,
    /// abi.h TF_DS_HIP_ABI_VERSION this binary was built for (0 on CUDA: no AMD kit)
    kit_abi: u32,
    /// SHA-256 (hex) of the AMD kit's manifest.json: the same code objects on every rank (zeros on CUDA)
    kit_sha256: [64]u8,
    /// model.Modalities as its tag (text 0, text_image 1)
    modalities: u8,
    plan: []const u8,
};

/// Opens a rank's collectives over its links (handshake first): the transport a HIP build plugs in. Errors here are
/// the transport's own (a refused handshake is Mismatch); the identity is the caller's and outlives the call only.
/// What a device-side transport binds its kernels from (tcp_proxy's coll.stage / coll.collect): the HIP build's AMD
/// kit; CUDA builds have none. The kit outlives the Coll (model.deinit closes the collective first).
pub const Kit = if (gpu.backend == .hip) gpu.Kit else void;
pub const Open = *const fn (gpa: std.mem.Allocator, io: std.Io, ctx: *const gpu.Context, kit: ?*const Kit, fan: *const Fan, id: *const Identity) anyerror!Coll;

test "a collective counts bytes by element type, passes its site, and refuses an exchange it lacks" {
    const Fake = struct {
        bytes: usize = 0,
        site: Site = .{ .phase = .test_only, .layer = 0 },
        fn gather(ptr: *anyopaque, _: gpu.Stream, site: Site, _: u64, _: u64, bytes: usize) Error!void {
            const f: *@This() = @ptrCast(@alignCast(ptr));
            f.bytes = bytes;
            f.site = site;
        }
        fn abort(_: *anyopaque, _: Reason) void {}
        fn close(_: *anyopaque) void {}
    };
    var f: Fake = .{};
    const vt: Coll.VTable = .{ .all_gather = Fake.gather, .abort = Fake.abort, .close = Fake.close };
    const c: Coll = .{ .ptr = &f, .vtable = &vt, .rank = 0, .world = 2, .capturable = false };
    try c.allGather(.at(.round_attn, 42), 0, 0, 6 * 5120, .f32, undefined);
    try std.testing.expectEqual(@as(usize, 6 * 5120 * 4), f.bytes);
    try std.testing.expectEqual(Phase.round_attn, f.site.phase);
    try std.testing.expectEqual(@as(u16, 42), f.site.layer);
    try c.allGather(.at(.round_head, 61), 0, 0, 7, .bf16, undefined);
    try std.testing.expectEqual(@as(usize, 14), f.bytes);
    try std.testing.expectError(error.Unsupported, c.exchange(.at(.two_pair, 0), 0, 1, 0, 1, .f32, 1, undefined));
    // a transport without the split gathers whole at gatherEnd
    const p = try c.gatherBegin(.at(.prompt_moe, 3), 0, 0, 10, .f32, undefined);
    try std.testing.expectEqual(@as(usize, 14), f.bytes);
    try c.gatherEnd(p, undefined);
    try std.testing.expectEqual(@as(usize, 40), f.bytes);
    try std.testing.expectEqual(Phase.prompt_moe, f.site.phase);
}


test "TF_DS_COLL_TIME: unset, empty or 0 is off; 1 is call, sync is sync; anything else is refused" {
    try std.testing.expectEqual(@as(?Timer.Mode, null), try Timer.parseMode(null));
    try std.testing.expectEqual(@as(?Timer.Mode, null), try Timer.parseMode(""));
    try std.testing.expectEqual(@as(?Timer.Mode, null), try Timer.parseMode("0"));
    try std.testing.expectEqual(@as(?Timer.Mode, .call), try Timer.parseMode("1"));
    try std.testing.expectEqual(@as(?Timer.Mode, .sync), try Timer.parseMode("sync"));
    try std.testing.expectError(error.BadSetting, Timer.parseMode("on"));
    try std.testing.expectError(error.BadSetting, Timer.parseMode("2"));
}

/// A transport that records every vtable call (kind, site, sizes, peer) in order.
const Recorder = struct {
    const Rec = struct { kind: u8, phase: Phase, layer: u16, send: u64, recv: u64, bytes: usize, rbytes: usize = 0, peer: u32 = 0 };
    log: [16]Rec = undefined,
    n: usize = 0,
    split: bool,
    fn of(ptr: *anyopaque) *Recorder {
        return @ptrCast(@alignCast(ptr));
    }
    fn gather(ptr: *anyopaque, _: gpu.Stream, site: Site, send: u64, recv: u64, bytes: usize) Error!void {
        const r = of(ptr);
        r.log[r.n] = .{ .kind = 'g', .phase = site.phase, .layer = site.layer, .send = send, .recv = recv, .bytes = bytes };
        r.n += 1;
    }
    fn begin(ptr: *anyopaque, _: gpu.Stream, site: Site, send: u64, recv: u64, bytes: usize) Error!void {
        const r = of(ptr);
        r.log[r.n] = .{ .kind = 'b', .phase = site.phase, .layer = site.layer, .send = send, .recv = recv, .bytes = bytes };
        r.n += 1;
    }
    fn end(ptr: *anyopaque, _: gpu.Stream) Error!void {
        const r = of(ptr);
        r.log[r.n] = .{ .kind = 'e', .phase = .test_only, .layer = 0, .send = 0, .recv = 0, .bytes = 0 };
        r.n += 1;
    }
    fn exchange(ptr: *anyopaque, _: gpu.Stream, site: Site, send: u64, sb: usize, recv: u64, rb: usize, peer: u32) Error!void {
        const r = of(ptr);
        r.log[r.n] = .{ .kind = 'x', .phase = site.phase, .layer = site.layer, .send = send, .recv = recv, .bytes = sb, .rbytes = rb, .peer = peer };
        r.n += 1;
    }
    fn abort(_: *anyopaque, _: Reason) void {}
    fn close(_: *anyopaque) void {}
    const split_vt: Coll.VTable = .{ .all_gather = gather, .exchange = exchange, .abort = abort, .close = close, .gather_begin = begin, .gather_end = end };
    const whole_vt: Coll.VTable = .{ .all_gather = gather, .exchange = exchange, .abort = abort, .close = close };
    fn coll(r: *Recorder) Coll {
        return .{ .ptr = r, .vtable = if (r.split) &split_vt else &whole_vt, .rank = 0, .world = 4, .capturable = false };
    }
    /// The sequence the engine issues: a round gather, a pair exchange, a split gather (begin, end).
    fn issue(c: Coll) !void {
        try c.allGather(.at(.round_attn, 7), 0x1000, 0x2000, 96, .f32, undefined);
        try c.exchange(.at(.two_pair, 1), 0x3000, 40, 0x4000, 32, .bf16, 2, undefined);
        const p = try c.gatherBegin(.at(.prompt_moe, 3), 0x5000, 0x6000, 10, .f32, undefined);
        try c.gatherEnd(p, undefined);
    }
};

var fake_drains: u64 = 0;
fn fakeDrain(_: gpu.Stream) Error!void {
    fake_drains += 1;
}

test "TF_DS_COLL_TIME off: the transport sees exactly the calls it saw before, and no timer line is logged" {
    // the calls with no timer, against the calls written straight to the vtable (the code before the timer)
    for ([_]bool{ false, true }) |split| {
        var a: Recorder = .{ .split = split };
        try Recorder.issue(a.coll());
        var b: Recorder = .{ .split = split };
        const vt = b.coll().vtable;
        try vt.all_gather(&b, undefined, .at(.round_attn, 7), 0x1000, 0x2000, 96 * 4);
        try vt.exchange.?(&b, undefined, .at(.two_pair, 1), 0x3000, 80, 0x4000, 64, 2);
        if (split) {
            try vt.gather_begin.?(&b, undefined, .at(.prompt_moe, 3), 0x5000, 0x6000, 40);
            try vt.gather_end.?(&b, undefined);
        } else try vt.all_gather(&b, undefined, .at(.prompt_moe, 3), 0x5000, 0x6000, 40);
        try std.testing.expectEqual(b.n, a.n);
        for (a.log[0..a.n], b.log[0..b.n]) |x, y| try std.testing.expectEqual(y, x);
        // and with the timer on (call mode) the transport still sees the same calls
        var c: Recorder = .{ .split = split };
        var t: Timer = .{ .mode = .call };
        var cc = c.coll();
        cc.timer = &t;
        try Recorder.issue(cc);
        try std.testing.expectEqual(b.n, c.n);
        for (c.log[0..c.n], b.log[0..b.n]) |x, y| try std.testing.expectEqual(y, x);
    }
    var buf: [512]u8 = undefined;
    try std.testing.expectEqual(@as(?[]const u8, null), Timer.line(null, &buf, 3, 0, "reply end"));
    var r: Recorder = .{ .split = true };
    try std.testing.expectEqual(@as(?*Timer, null), r.coll().timer);
}

test "TF_DS_COLL_TIME: sums by kind into the pass and the rounds, sync drains around each call, the line's shape" {
    var r: Recorder = .{ .split = true };
    var t: Timer = .{ .mode = .sync, .drain = fakeDrain };
    var c = r.coll();
    c.timer = &t;
    fake_drains = 0;
    try Recorder.issue(c); // rounds: a gather, an exchange, a split gather
    try std.testing.expectEqual(@as(u64, 8), fake_drains); // before and after each of the four calls
    t.in_pass = true;
    try Recorder.issue(c);
    try Recorder.issue(c);
    t.in_pass = false;
    t.round_count = 1;
    try std.testing.expectEqual([3]u64{ 1, 1, 1 }, t.rounds.calls);
    try std.testing.expectEqual([3]u64{ 2, 2, 2 }, t.pass.calls);
    try std.testing.expectEqual(@as(u64, 0), t.begun_ns);
    // the line's exact shape, from fixed sums
    t.pass = .{ .calls = .{ 120, 0, 61 }, .ns = .{ 2_500_000, 0, 1_250_400 } };
    t.rounds = .{ .calls = .{ 0, 4890, 0 }, .ns = .{ 0, 145_000_000, 0 } };
    t.round_count = 30;
    var buf: [512]u8 = undefined;
    const s = Timer.line(&t, &buf, 41, 0, "reply end").?;
    try std.testing.expectEqualStrings("lanes: step 41 rank 0 coll time (reply end, sync): pass 3.750 ms (all_gather 120 2.500, exchange 0 0.000, prompt_gather 61 1.250); rounds 30 145.000 ms (all_gather 0 0.000, exchange 4890 145.000, prompt_gather 0 0.000), 4.833 ms a round", s);
    // a line clears the sums: the next one counts from there (no round: 0 ms a round)
    const z = Timer.line(&t, &buf, 42, 0, "pass start").?;
    try std.testing.expectEqualStrings("lanes: step 42 rank 0 coll time (pass start, sync): pass 0.000 ms (all_gather 0 0.000, exchange 0 0.000, prompt_gather 0 0.000); rounds 0 0.000 ms (all_gather 0 0.000, exchange 0 0.000, prompt_gather 0 0.000), 0.000 ms a round", z);
    // call mode never drains
    fake_drains = 0;
    t.mode = .call;
    try Recorder.issue(c);
    try std.testing.expectEqual(@as(u64, 0), fake_drains);
}
