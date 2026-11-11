//! TF_COLL_KIND=rdma: tcp_proxy's slot collective, each exchange one RC write with immediate (COLLECTIVES-DESIGN 3.6).
const std = @import("std");
const gpu = @import("gpu");
const coll = @import("coll.zig");
const Fan = @import("fan.zig").Fan;
const tcp_coll = @import("tcp_coll.zig");
const tcp_proxy = @import("tcp_proxy.zig");
pub const nic = @import("rdma_nic.zig");
pub const Core = @import("rdma_core.zig").Core;

const c = std.c;
const Verbs = nic.Verbs;
const Nic = nic.Nic;
pub const Shared = tcp_proxy.Shared;
pub const max_slots = tcp_proxy.max_slots;
pub const Error = nic.Error;
const StageK = gpu.Kernel("coll.stage", tcp_proxy.params.Stage);
const CollectK = gpu.Kernel("coll.collect", tcp_proxy.params.Collect);

/// The same slab, kernels, generation protocol and fault contract as tcp_proxy; graph capture is unchanged.
pub const RdmaProxy = struct {
    gpa: std.mem.Allocator,
    d: *const gpu.Driver,
    rank: u32,
    host: *tcp_coll.TcpColl,
    verbs: Verbs,
    nic: Nic,
    peer: nic.Info,
    epoch: [16]u8,
    slab: gpu.HostBuffer,
    shared: *Shared,
    dev: u64, // the slab's device address
    tx_off: u64,
    rx_off: u64,
    nslot: u32,
    slot_bytes: u64,
    deadline_ticks: u64,
    progress_ns: u64,
    stage: StageK,
    collect: CollectK,
    proxy_cpu: i32,
    thread: ?std.Thread = null,
    stop: std.atomic.Value(u32) = .init(0),
    /// the proxy's failure (coll.Error as a code), 0 while healthy
    failed: std.atomic.Value(u32) = .init(0),
    /// gathers the proxy has completed (diagnostics)
    exchanged: std.atomic.Value(u64) = .init(0),
    begun: ?Begun = null,

    const Begun = struct { host: bool, recv: u64, bytes: usize };
    const vtable: coll.Coll.VTable = .{ .all_gather = allGather, .abort = abort, .close = close, .gather_begin = gatherBegin, .gather_end = gatherEnd, .check = check, .counts = counts };

    fn counts(ptr: *anyopaque) coll.Coll.Counts {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        return .{ .staged = @atomicLoad(u64, &t.shared.seq, .acquire), .cable1 = t.exchanged.load(.acquire), .host = t.host.w.seq };
    }

    fn check(ptr: *anyopaque) coll.Error!void {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
    }

    fn coll_(t: *RdmaProxy) coll.Coll {
        return .{ .ptr = t, .vtable = &vtable, .rank = t.rank, .world = 2, .capturable = true };
    }

    fn status(t: *RdmaProxy) ?coll.Error {
        const f = t.failed.load(.acquire);
        if (f != 0) return switch (f) {
            1 => error.PeerLost,
            2 => error.Timeout,
            3 => error.Mismatch,
            4 => error.Aborted,
            else => error.CollectiveFailed,
        };
        if (@atomicLoad(u32, &t.shared.abort, .acquire) != 0) return error.Timeout; // a device wait gave up
        return null;
    }

    fn poison(t: *RdmaProxy, code: u32) void {
        _ = t.failed.cmpxchgStrong(0, code, .acq_rel, .acquire);
        tcp_proxy.raiseAbort(t.shared);
        t.nic.toErr();
        t.host.w.abort();
    }

    fn proxyLoop(t: *RdmaProxy) void {
        tcp_proxy.TcpProxy.pin(t.proxy_cpu);
        var core: Core = .{
            .shared = t.shared,
            .tx = t.slab.bytes[t.tx_off..][0 .. t.nslot * t.slot_bytes],
            .rx = t.slab.bytes[t.rx_off..][0 .. t.nslot * t.slot_bytes],
            .nslot = t.nslot,
            .slot_bytes = t.slot_bytes,
            .nic = &t.nic,
            .peer_rx = t.peer.rx_base,
            .peer_rkey = t.peer.rkey,
            .epoch = t.epoch,
            .stop = &t.stop,
            .exchanged = &t.exchanged,
            .progress_ns = t.progress_ns,
            .peer_fd = t.host.w.fds[0],
            .rank = t.rank,
            .warn_ns = t.host.gpu_warn_ns,
        };
        const code = core.run();
        if (code != 0) t.poison(code);
    }

    fn allGather(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) coll.Error!void {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        if (t.begun != null) return t.misuse("an all-gather while a begun gather is pending");
        try gatherBegin(ptr, stream, site, send, recv, bytes);
        return gatherEnd(ptr, stream);
    }

    fn misuse(t: *RdmaProxy, what: []const u8) coll.Error {
        std.log.err("collectives: {s}", .{what});
        t.poison(5);
        return error.CollectiveFailed;
    }

    /// The stage kernel (this rank's block into its slot) or, for a gather larger than a slot, tcp_host's begin.
    fn gatherBegin(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) coll.Error!void {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
        if (t.begun != null) return t.misuse("a gather begun while another is pending");
        if (bytes > nic.capacity(t.slot_bytes)) {
            const capturing = stream.capturing() catch return error.CollectiveFailed;
            if (capturing) {
                std.log.err("collectives: a {d}-byte gather inside a graph exceeds what a {d}-byte slot carries (TF_COLL_SLOT_KIB)", .{ bytes, t.slot_bytes });
                t.poison(5);
                return error.CollectiveFailed;
            }
            try t.host.coll_().vtable.gather_begin.?(t.host, stream, site, send, recv, bytes);
            t.begun = .{ .host = true, .recv = recv, .bytes = bytes };
            return;
        }
        t.begun = .{ .host = false, .recv = recv, .bytes = bytes };
        if (bytes == 0) return;
        const site_word = (@as(u32, @intFromEnum(site.phase)) << 16) | site.layer;
        const cfg: gpu.Config = .{ .grid = .{ .x = tcp_proxy.TcpProxy.blocks(bytes), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } };
        t.stage.launch(stream, cfg, .{ .s = t.dev, .send = send, .tx = t.dev + t.tx_off, .own = recv + @as(u64, t.rank) * bytes, .bytes = bytes, .slot_bytes = t.slot_bytes, .nslot = t.nslot, .site = site_word, .deadline = t.deadline_ticks }) catch {
            t.poison(5);
            return error.CollectiveFailed;
        };
    }

    /// The collect kernel (waits for the proxy, then the peer's block into `recv`), or tcp_host's end.
    fn gatherEnd(ptr: *anyopaque, stream: gpu.Stream) coll.Error!void {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        const b = t.begun orelse return t.misuse("a gather ended that was never begun");
        t.begun = null;
        if (b.host) return t.host.coll_().vtable.gather_end.?(t.host, stream);
        if (t.status()) |e| return e;
        if (b.bytes == 0) return;
        const peer: u64 = 1 - t.rank;
        const cfg: gpu.Config = .{ .grid = .{ .x = tcp_proxy.TcpProxy.blocks(b.bytes), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } };
        t.collect.launch(stream, cfg, .{ .s = t.dev, .rx = t.dev + t.rx_off, .peer = b.recv + peer * b.bytes, .bytes = b.bytes, .slot_bytes = t.slot_bytes, .nslot = t.nslot, .deadline = t.deadline_ticks }) catch {
            t.poison(5);
            return error.CollectiveFailed;
        };
    }

    fn abort(ptr: *anyopaque, reason: coll.Reason) void {
        _ = reason;
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        t.poison(4);
        t.host.coll_().vtable.abort(t.host, .cancelled);
    }

    fn close(ptr: *anyopaque) void {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        t.stop.store(1, .release);
        @atomicStore(u32, &t.shared.abort, 1, .release); // ends any device wait
        t.nic.toErr(); // posted work flushes; the loop leaves
        if (t.thread) |th| th.join();
        // queued kernels see the abort word and leave the slab alone; the MRs go after they drained and the QP is gone
        _ = t.d.api.hipDeviceSynchronize();
        t.nic.close();
        t.verbs.close();
        t.slab.free();
        t.host.coll_().vtable.close(t.host);
        const gpa = t.gpa;
        gpa.destroy(t);
    }
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const gpu.Context, kit: *const gpu.Kit, fan: *const Fan, id: *const coll.Identity) anyerror!coll.Coll {
    _ = io;
    const d = ctx.d;
    const nslot: u32 = @intCast(try tcp_proxy.envU("TF_COLL_SLOTS", 8));
    if (nslot < 2 or nslot > max_slots) {
        std.log.err("collectives: TF_COLL_SLOTS={d}: kind=rdma needs 2 to {d} slots (receive-slot reuse is safe only transitively)", .{ nslot, max_slots });
        return error.BadSetting;
    }
    const slot_kib = try tcp_proxy.envU("TF_COLL_SLOT_KIB", 4352); // 4.25 MiB: a 16-row head gather (4,136,960 B) plus the trailer fits
    const slot_bytes = std.mem.alignForward(u64, slot_kib << 10, 4096);
    const proxy_cpu: i32 = @intCast(try tcp_proxy.envU("TF_COLL_PROXY_CPU", 6));
    const stripe_kib = try tcp_proxy.envU("TF_COLL_PROXY_STRIPE_KIB", 0);
    if (stripe_kib != 0) {
        std.log.err("collectives: TF_COLL_PROXY_STRIPE_KIB={d}: kind=rdma moves every slot gather over one QP (a second thread breaks the one-sequence rule)", .{stripe_kib});
        return error.BadSetting;
    }
    const gid_index = nic.parseGidIndex(if (c.getenv("TF_RDMA_GID_INDEX")) |v| std.mem.span(v) else null) catch {
        std.log.err("collectives: TF_RDMA_GID_INDEX: a decimal GID index in 0..255", .{});
        return error.BadSetting;
    };
    var vb = try Verbs.open(null);
    errdefer vb.close();
    var name_buf: [64]u8 = undefined;
    const dev_name = try nic.deviceName(&vb, &name_buf, id.rank);
    // the device name is this rank's own (the two boxes may differ), so it is logged, not compared
    var extra_buf: [128]u8 = undefined;
    const extra = try std.fmt.bufPrint(&extra_buf, "kind=rdma gid={d} tc=0 slots={d} slot_kib={d} stripe=0", .{ gid_index, nslot, slot_kib });
    const se = try tcp_coll.handshake(fan, id, extra);
    const host = try tcp_coll.openHost(gpa, d, id.rank, se, 0);
    errdefer host.coll_().vtable.close(host);
    const stage = try StageK.bind(kit, &.{});
    const collect = try CollectK.bind(kit, &.{});
    const caps = try ctx.caps();
    const tx_off: u64 = 4096;
    const rx_off: u64 = tx_off + nslot * slot_bytes;
    var slab = try gpu.HostBuffer.allocFlags(d, rx_off + nslot * slot_bytes, .{ .mapped = true, .coherent = true });
    errdefer slab.free();
    @memset(slab.bytes[0..@sizeOf(Shared)], 0);
    const t = try gpa.create(RdmaProxy);
    errdefer gpa.destroy(t);
    t.* = .{
        .gpa = gpa,
        .d = d,
        .rank = id.rank,
        .host = host,
        .verbs = vb,
        .nic = undefined,
        .peer = undefined,
        .epoch = se.epoch,
        .slab = slab,
        .shared = @ptrCast(@alignCast(slab.bytes.ptr)),
        .dev = try slab.device(),
        .tx_off = tx_off,
        .rx_off = rx_off,
        .nslot = nslot,
        .slot_bytes = slot_bytes,
        .deadline_ticks = tcp_proxy.deadlineTicks(caps.wall_clock_hz, se.settings.gpu_wait_ns),
        .progress_ns = se.settings.opt.progress_ns,
        .stage = stage,
        .collect = collect,
        .proxy_cpu = proxy_cpu,
    };
    t.nic = try Nic.open(&t.verbs, dev_name, gid_index, slab.bytes[tx_off..][0 .. nslot * slot_bytes], slab.bytes[rx_off..][0 .. nslot * slot_bytes]);
    errdefer t.nic.close();
    t.peer = try nic.exchangeInfo(fan, t.nic.info(nslot, slot_bytes));
    try t.nic.connect(t.peer);
    try nic.startBarrier(fan);
    t.thread = try std.Thread.spawn(.{}, RdmaProxy.proxyLoop, .{t});
    std.log.info("collectives: rdma_proxy on {s} gid={d} (mtu {d}, qp {d} -> peer qp {d}, rkey {x}) with {d} slots of {d} KiB (coherent slab {d} MiB, {d}-byte trailer), proxy thread on CPU {d}, progress deadline {d} ms; larger gathers through tcp_host", .{
        dev_name, gid_index, t.nic.active_mtu, t.nic.qp.qp_num, t.peer.qpn, t.peer.rkey, nslot, slot_bytes >> 10, slab.bytes.len >> 20, nic.trailer_bytes, proxy_cpu, t.progress_ns / std.time.ns_per_ms,
    });
    return t.coll_();
}

test {
    _ = @import("rdma_proxy_test.zig");
}
