//! rdma_proxy's tests: the trailer arithmetic, settings, and the slab protocol over two QPs on one HCA (loopback).
const std = @import("std");
const coll = @import("coll.zig");
const wire = @import("tcp_wire.zig");
const nic = @import("rdma_nic.zig");
const Core = @import("rdma_core.zig").Core;
const Shared = @import("tcp_proxy.zig").Shared;

const c = std.c;
const testing = std.testing;
const Verbs = nic.Verbs;
const Nic = nic.Nic;

test "trailer placement and the slot capacity" {
    try testing.expectEqual(@as(u64, 0), nic.trailerAt(0));
    try testing.expectEqual(@as(u64, 64), nic.trailerAt(1));
    try testing.expectEqual(@as(u64, 20480), nic.trailerAt(20480));
    try testing.expectEqual(@as(u64, 20544), nic.wireLen(20480));
    try testing.expectEqual(@as(u64, 4_456_448 - 64), nic.capacity(4_456_448));
    try testing.expect(nic.wireLen(16 * 64640 * 4) <= 4_456_448); // the 16-row head gather fits the default slot
    try testing.expect(nic.wireLen(nic.capacity(4_456_448)) == 4_456_448);
}

test "the GID index defaults to 3 and refuses malformed settings" {
    try testing.expectEqual(@as(u8, 3), try nic.parseGidIndex(null));
    try testing.expectEqual(@as(u8, 5), try nic.parseGidIndex("5"));
    for ([_][]const u8{ "", "-1", "+3", " 3", "0x3", "256" }) |bad| try testing.expectError(error.BadSetting, nic.parseGidIndex(bad));
}

test "TF_RDMA_DEVICES: one name for both ranks, or one per rank" {
    const env = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    defer _ = env.unsetenv("TF_RDMA_DEVICES");
    var v: Verbs = undefined;
    var buf: [64]u8 = undefined;
    _ = env.setenv("TF_RDMA_DEVICES", "mlx5_0,mlx5_1", 1);
    try testing.expectEqualStrings("mlx5_0", try nic.deviceName(&v, &buf, 0));
    try testing.expectEqualStrings("mlx5_1", try nic.deviceName(&v, &buf, 1));
    _ = env.setenv("TF_RDMA_DEVICES", "mlx5_2", 1);
    try testing.expectEqualStrings("mlx5_2", try nic.deviceName(&v, &buf, 1));
}

test "the Info a peer reads is a plain extern struct" {
    try testing.expectEqual(@as(usize, 64), @sizeOf(nic.Info));
    try testing.expectEqual(@as(usize, 40), @offsetOf(nic.Info, "rx_base"));
}

/// A host thread doing what coll.stage and coll.collect do on a GPU, with ways to misbehave.
const FakeGpu = struct {
    shared: *Shared,
    tx: []u8,
    rx: []u8,
    nslot: u32,
    slot_bytes: u64,
    rank: u32,
    gathers: usize,
    bad: usize = 0,
    /// stage only, never collect (the peer's slots then fill up and its stage must wait)
    hold_collect_from: usize = std.math.maxInt(usize),
    /// a slow collector: this long between done and consumed
    collect_delay_ns: u64 = 0,
    /// at this gather, a different site (or bytes) from the peer's
    wrong_site_at: usize = 0,
    wrong_bytes_at: usize = 0,
    fixed_bytes: ?usize = null,
    timed_out: bool = false,
    /// ready -> done per gather (ns), for the timing test
    rtt: ?[]u64 = null,

    fn pattern(q: u64, rank: u32, i: usize) u8 {
        return @truncate((q *% 131) ^ (@as(u64, rank) << 7) ^ i);
    }

    fn bytesOf(f: *const FakeGpu, q: u64) usize {
        return f.fixed_bytes orelse 1000 + (q % 7) * 4096;
    }

    fn run(f: *FakeGpu) void {
        for (1..f.gathers + 1) |qi| {
            const q: u64 = qi;
            const slot: usize = @intCast(q % f.nslot);
            const bytes: usize = f.bytesOf(q) + @as(usize, if (qi == f.wrong_bytes_at) 64 else 0);
            const t0 = wire.monotonicNs();
            while (q > f.nslot and @atomicLoad(u64, &f.shared.slot[slot].consumed, .acquire) + f.nslot < q) {
                if (@atomicLoad(u32, &f.shared.abort, .acquire) != 0 or wire.monotonicNs() - t0 > 2 * std.time.ns_per_s) {
                    f.timed_out = true;
                    return;
                }
                std.atomic.spinLoopHint();
            }
            const tx = f.tx[slot * f.slot_bytes ..][0..bytes];
            for (tx, 0..) |*b, i| b.* = pattern(q, f.rank, i);
            f.shared.slot[slot].bytes = bytes;
            f.shared.slot[slot].site = if (qi == f.wrong_site_at) 9 else 7;
            const t1 = wire.monotonicNs();
            @atomicStore(u64, &f.shared.slot[slot].ready, q, .release);
            if (qi >= f.hold_collect_from) continue;
            while (@atomicLoad(u64, &f.shared.slot[slot].done, .acquire) != q) {
                if (@atomicLoad(u32, &f.shared.abort, .acquire) != 0) return;
                std.atomic.spinLoopHint();
            }
            if (f.rtt) |r| r[qi - 1] = wire.monotonicNs() - t1;
            if (f.collect_delay_ns > 0) {
                const t2 = wire.monotonicNs();
                while (wire.monotonicNs() - t2 < f.collect_delay_ns) std.atomic.spinLoopHint();
            }
            const rx = f.rx[slot * f.slot_bytes ..][0..bytes];
            for (rx, 0..) |b, i| f.bad += @intFromBool(b != pattern(q, 1 - f.rank, i));
            @atomicStore(u64, &f.shared.slot[slot].consumed, q, .release);
        }
    }
};

const TestRank = struct {
    shared: Shared = std.mem.zeroes(Shared),
    tx: []u8,
    rx: []u8,
    nic: Nic,
    core: Core = undefined,
    stop: std.atomic.Value(u32) = .init(0),
    exchanged: std.atomic.Value(u64) = .init(0),
    code: u32 = 0,
    alive: bool = true,

    fn runCore(r: *TestRank) void {
        r.code = r.core.run();
    }

    fn kill(r: *TestRank) void {
        if (r.alive) r.nic.close();
        r.alive = false;
    }
};

const Pair = struct {
    v: Verbs,
    rk: [2]*TestRank,
    nslot: u32,
    slot_bytes: u64,

    fn close(p: *Pair) void {
        for (p.rk) |r| {
            r.kill();
            std.heap.page_allocator.free(r.tx);
            std.heap.page_allocator.free(r.rx);
            testing.allocator.destroy(r);
        }
        p.v.close();
    }

    fn gpus(p: *Pair, gathers: usize) [2]FakeGpu {
        var g: [2]FakeGpu = undefined;
        for (&g, p.rk, 0..) |*f, r, i| f.* = .{ .shared = &r.shared, .tx = r.tx, .rx = r.rx, .nslot = p.nslot, .slot_bytes = p.slot_bytes, .rank = @intCast(i), .gathers = gathers };
        return g;
    }
};

/// Two ranks' slabs (page memory) and QPs on TF_RDMA_TEST_DEVICE, connected to each other; null without a device.
fn testPair(nslot: u32, slot_bytes: u64, progress_ns: u64) !?Pair {
    const dev = c.getenv("TF_RDMA_TEST_DEVICE") orelse return null;
    const gid_index = try nic.parseGidIndex(if (c.getenv("TF_RDMA_GID_INDEX")) |v| std.mem.span(v) else null);
    var p: Pair = .{ .v = try Verbs.open(null), .rk = undefined, .nslot = nslot, .slot_bytes = slot_bytes };
    errdefer p.v.close();
    const epoch: [16]u8 = @splat(3);
    for (&p.rk) |*slot| {
        const r = try testing.allocator.create(TestRank);
        const n: usize = @intCast(nslot * slot_bytes);
        r.* = .{ .tx = try std.heap.page_allocator.alloc(u8, n), .rx = try std.heap.page_allocator.alloc(u8, n), .nic = undefined };
        @memset(r.rx, 0);
        r.nic = try Nic.open(&p.v, std.mem.span(dev), gid_index, r.tx, r.rx);
        slot.* = r;
    }
    for (p.rk, 0..) |r, i| {
        const peer = p.rk[1 - i];
        try r.nic.connect(peer.nic.info(nslot, slot_bytes));
        r.core = .{ .shared = &r.shared, .tx = r.tx, .rx = r.rx, .nslot = nslot, .slot_bytes = slot_bytes, .nic = &r.nic, .peer_rx = peer.nic.rx_base, .peer_rkey = peer.nic.mr_rx.rkey, .epoch = epoch, .stop = &r.stop, .exchanged = &r.exchanged, .progress_ns = progress_ns, .rank = @intCast(i) };
    }
    return p;
}

fn runBoth(p: *Pair, g: *[2]FakeGpu) !void {
    const c0 = try std.Thread.spawn(.{}, TestRank.runCore, .{p.rk[0]});
    const c1 = try std.Thread.spawn(.{}, TestRank.runCore, .{p.rk[1]});
    const g1 = try std.Thread.spawn(.{}, FakeGpu.run, .{&g[1]});
    g[0].run();
    g1.join();
    for (p.rk) |r| r.stop.store(1, .release);
    for (p.rk) |r| r.nic.toErr(); // a core waiting for a completion that never comes leaves at once
    c0.join();
    c1.join();
}

test "rdma loopback: the slab protocol moves every gather through reused slots, byte-verified, and stops cleanly" {
    var p = (try testPair(4, 32768, 2 * std.time.ns_per_s)) orelse return error.SkipZigTest;
    defer p.close();
    var g = p.gpus(2000);
    try runBoth(&p, &g);
    try testing.expectEqual(@as(usize, 0), g[0].bad + g[1].bad);
    for (p.rk) |r| {
        try testing.expectEqual(@as(u32, 0), r.code);
        try testing.expectEqual(@as(u64, 2000), r.exchanged.load(.acquire));
        // TX reuse: a slot was re-staged only after its send completion (one per gather, all seen)
        try testing.expectEqual(@as(u64, 2000), r.core.sends);
        try testing.expectEqual(@as(u64, 2000), r.core.recvs);
    }
}

test "rdma loopback: RX reuse with NSLOT=2, a slow collector on one rank and a fast peer: nothing is overwritten" {
    var p = (try testPair(2, 32768, 2 * std.time.ns_per_s)) orelse return error.SkipZigTest;
    defer p.close();
    var g = p.gpus(400);
    g[1].collect_delay_ns = 300 * std.time.ns_per_us;
    try runBoth(&p, &g);
    try testing.expectEqual(@as(usize, 0), g[0].bad + g[1].bad);
    for (p.rk) |r| {
        try testing.expectEqual(@as(u32, 0), r.code);
        try testing.expectEqual(@as(u64, 400), r.exchanged.load(.acquire));
    }
}

test "rdma loopback: a slot is not reused before it was consumed: a stalled collect stops the stage, not the data" {
    var p = (try testPair(4, 32768, 2 * std.time.ns_per_s)) orelse return error.SkipZigTest;
    defer p.close();
    var g = p.gpus(50);
    g[1].hold_collect_from = 10; // rank 1 stops collecting at gather 10: its stage must stall at gather 14
    const c0 = try std.Thread.spawn(.{}, TestRank.runCore, .{p.rk[0]});
    const c1 = try std.Thread.spawn(.{}, TestRank.runCore, .{p.rk[1]});
    const g0 = try std.Thread.spawn(.{}, FakeGpu.run, .{&g[0]});
    g[1].run();
    try testing.expect(g[1].timed_out); // waited for a slot that was never released
    for (p.rk) |r| r.stop.store(1, .release);
    @atomicStore(u32, &p.rk[0].shared.abort, 1, .release);
    for (p.rk) |r| r.nic.toErr();
    g0.join();
    c0.join();
    c1.join();
    try testing.expectEqual(@as(usize, 0), g[0].bad + g[1].bad); // nothing overwritten while still owned
    try testing.expect(p.rk[1].exchanged.load(.acquire) <= 13);
}

test "rdma loopback: a different site or size from the peer fails both ranks with Mismatch" {
    for ([_]u8{ 0, 1 }) |which| {
        var p = (try testPair(4, 32768, 2 * std.time.ns_per_s)) orelse return error.SkipZigTest;
        defer p.close();
        var g = p.gpus(30);
        if (which == 0) g[0].wrong_site_at = 12 else g[0].wrong_bytes_at = 12;
        try runBoth(&p, &g);
        try testing.expectEqual(@as(u32, 3), p.rk[0].code);
        try testing.expectEqual(@as(u32, 3), p.rk[1].code);
        try testing.expect(p.rk[0].exchanged.load(.acquire) == 11 and p.rk[1].exchanged.load(.acquire) == 11);
        try testing.expect(@atomicLoad(u32, &p.rk[0].shared.abort, .acquire) != 0);
    }
}

test "rdma loopback: a peer that stops staging fails the proxy at the progress deadline (Timeout), after its own send completed" {
    var p = (try testPair(4, 32768, 500 * std.time.ns_per_ms)) orelse return error.SkipZigTest;
    defer p.close();
    // rank 0 stages gather 1; rank 1 never does
    p.rk[0].shared.slot[1].bytes = 4096;
    p.rk[0].shared.slot[1].site = (@as(u32, @intFromEnum(coll.Phase.prompt_engram)) << 16) | 3;
    @atomicStore(u64, &p.rk[0].shared.slot[1].ready, 1, .release);
    const t0 = wire.monotonicNs();
    const code = p.rk[0].core.run();
    const waited = wire.monotonicNs() - t0;
    try testing.expectEqual(@as(u32, 2), code);
    try testing.expectEqual(@as(u64, 1), p.rk[0].core.sends);
    try testing.expect(waited >= 500 * std.time.ns_per_ms and waited < 5 * std.time.ns_per_s);
    try testing.expectEqual(@as(u32, 1), @atomicLoad(u32, &p.rk[0].shared.abort, .acquire));
}

test "rdma loopback: a dead peer (its QP gone) fails the write with RETRY_EXC: PeerLost from the NIC" {
    var p = (try testPair(4, 32768, 8 * std.time.ns_per_s)) orelse return error.SkipZigTest;
    defer p.close();
    p.rk[1].kill();
    p.rk[0].shared.slot[1].bytes = 20480;
    p.rk[0].shared.slot[1].site = (@as(u32, @intFromEnum(coll.Phase.round_attn)) << 16) | 3;
    @atomicStore(u64, &p.rk[0].shared.slot[1].ready, 1, .release);
    const t0 = wire.monotonicNs();
    const code = p.rk[0].core.run();
    const waited = wire.monotonicNs() - t0;
    std.debug.print("rdma loopback: dead peer surfaced after {d} ms (qp timeout {d})\n", .{ waited / std.time.ns_per_ms, try nic.qpTimeout() });
    try testing.expectEqual(@as(u32, 1), code);
    try testing.expect(waited < 6 * std.time.ns_per_s);
}

test "rdma loopback: a stalled peer whose control socket closes is PeerLost within milliseconds" {
    var p = (try testPair(4, 32768, 4 * std.time.ns_per_s)) orelse return error.SkipZigTest;
    defer p.close();
    var sv: [2]c.fd_t = undefined;
    if (c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &sv) != 0) return error.SocketFailed;
    defer _ = c.close(sv[0]);
    p.rk[0].core.peer_fd = sv[0];
    p.rk[0].shared.slot[1].bytes = 4096;
    p.rk[0].shared.slot[1].site = 7;
    @atomicStore(u64, &p.rk[0].shared.slot[1].ready, 1, .release);
    const Closer = struct {
        fn run(fd: c.fd_t) void {
            var d: c.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_ms };
            _ = c.nanosleep(&d, null);
            _ = c.close(fd);
        }
    };
    const th = try std.Thread.spawn(.{}, Closer.run, .{sv[1]});
    const t0 = wire.monotonicNs();
    const code = p.rk[0].core.run();
    const waited = wire.monotonicNs() - t0;
    th.join();
    try testing.expectEqual(@as(u32, 1), code);
    try testing.expect(waited >= 100 * std.time.ns_per_ms and waited < 2 * std.time.ns_per_s);
}

test "rdma loopback: 20 KiB round trips, ready to done (hairpin through one HCA: slower than two boxes)" {
    var p = (try testPair(8, 32768, 2 * std.time.ns_per_s)) orelse return error.SkipZigTest;
    defer p.close();
    const n = 3000;
    var g = p.gpus(n);
    const rtt = try testing.allocator.alloc(u64, n);
    defer testing.allocator.free(rtt);
    for (&g) |*f| f.fixed_bytes = 20480;
    g[0].rtt = rtt;
    try runBoth(&p, &g);
    try testing.expectEqual(@as(usize, 0), g[0].bad + g[1].bad);
    const warm = 500;
    const v = rtt[warm..];
    std.mem.sort(u64, v, {}, std.sort.asc(u64));
    var sum: u64 = 0;
    for (v) |x| sum += x;
    std.debug.print("rdma loopback 20480 B: n={d} p50={d:.1} p99={d:.1} min={d:.1} max={d:.1} mean={d:.1} us (ready -> done on rank 0)\n", .{
        v.len, @as(f64, @floatFromInt(v[v.len / 2])) / 1e3, @as(f64, @floatFromInt(v[v.len * 99 / 100])) / 1e3, @as(f64, @floatFromInt(v[0])) / 1e3, @as(f64, @floatFromInt(v[v.len - 1])) / 1e3, @as(f64, @floatFromInt(sum / v.len)) / 1e3,
    });
}
