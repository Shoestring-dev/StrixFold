//! rdma_proxy's loop on the slab, GPU-free: ready, one write with immediate, both completions, trailer check, done.
const std = @import("std");
const nic = @import("rdma_nic.zig");
const wire = @import("tcp_wire.zig");
const tcp_proxy = @import("tcp_proxy.zig");

const c = std.c;
const vabi = nic.vabi;
pub const Shared = tcp_proxy.Shared;

/// run returns a failure code (0 clean stop, 1 PeerLost, 2 Timeout, 3 Mismatch, 5 verbs) after raising the abort word.
pub const Core = struct {
    shared: *Shared,
    tx: []u8, // nslot * slot_bytes
    rx: []u8,
    nslot: u32,
    slot_bytes: u64,
    nic: *nic.Nic,
    peer_rx: u64,
    peer_rkey: u32,
    epoch: [16]u8,
    stop: *std.atomic.Value(u32),
    exchanged: *std.atomic.Value(u64),
    /// no completion for this long after a write was posted: Timeout
    progress_ns: u64 = 2 * std.time.ns_per_s,
    /// a connected socket to the peer to peek at while waiting (-1: none): its end of stream is PeerLost at once
    peer_fd: c.fd_t = -1,
    /// for the log lines: this rank, and an exchange longer than this is logged (0: never)
    rank: u32 = 0,
    warn_ns: u64 = 0,
    /// diagnostics: completions seen
    sends: u64 = 0,
    recvs: u64 = 0,

    pub fn run(k: *Core) u32 {
        var q: u64 = 1;
        var idle_since: u64 = 0;
        // one completion per poll: the peer's write of q + 1 may already be in, and that one belongs to the next turn
        var wc: [1]vabi.Wc = undefined;
        while (k.stop.load(.acquire) == 0) {
            const slot: usize = @intCast(q % k.nslot);
            const r = @atomicLoad(u64, &k.shared.slot[slot].ready, .acquire);
            if (r != q) {
                if (r > q) { // a generation ahead of the one expected: stale or duplicated
                    std.log.warn("collectives: rank {d} proxy expected gather {d} in slot {d} and found gather {d}", .{ k.rank, q, slot, r });
                    return k.fail(3);
                }
                if (@atomicLoad(u32, &k.shared.abort, .acquire) != 0) {
                    tcp_proxy.deviceGaveUp(k.shared, k.rank, q, k.nslot, "waiting for its GPU to stage gather");
                    return k.fail(2);
                }
                if (idle_since == 0) idle_since = wire.monotonicNs();
                if (wire.monotonicNs() - idle_since > 20 * std.time.ns_per_ms) std.Thread.yield() catch {} else std.atomic.spinLoopHint();
                continue;
            }
            idle_since = 0;
            const bytes: u64 = k.shared.slot[slot].bytes;
            const site: u32 = @truncate(k.shared.slot[slot].site);
            if (nic.wireLen(bytes) > k.slot_bytes) {
                std.log.warn("collectives: rank {d} proxy: gather {d} says {d} bytes, more than a slot carries ({d})", .{ k.rank, q, bytes, nic.capacity(k.slot_bytes) });
                return k.fail(3);
            }
            const base: usize = slot * @as(usize, @intCast(k.slot_bytes));
            const at: usize = @intCast(nic.trailerAt(bytes));
            const tr: *nic.Trailer = @ptrCast(@alignCast(k.tx[base + at ..].ptr));
            tr.* = .{ .magic = nic.trailer_magic, .epoch = k.epoch, .q = q, .bytes = bytes, .site = site, .rank = k.rank };
            const len: usize = @intCast(nic.wireLen(bytes));
            const t0 = wire.monotonicNs();
            k.nic.postWrite(k.tx[base..][0..len], k.peer_rx + base, k.peer_rkey, q) catch {
                k.logExchange(q, site, bytes, t0, "ibv_post_send failed");
                return k.fail(5);
            };
            const outcome = k.completions(q, site, bytes, len, t0, &wc);
            if (outcome != 0) return outcome;
            const got: *const nic.Trailer = @ptrCast(@alignCast(k.rx[base + at ..].ptr));
            if (got.magic != nic.trailer_magic or !std.mem.eql(u8, &got.epoch, &k.epoch) or got.q != q or got.bytes != bytes or got.site != site or got.rank != 1 - k.rank) {
                std.log.warn("collectives: rank {d} proxy trailer mismatch at gather {d}: this rank's site {d}/{d} bytes {d}, the peer's (rank {d}) gather {d} site {d}/{d} bytes {d}{s}", .{
                    k.rank, q, site >> 16, site & 0xffff, bytes, got.rank, got.q, got.site >> 16, got.site & 0xffff, got.bytes,
                    if (got.magic != nic.trailer_magic or !std.mem.eql(u8, &got.epoch, &k.epoch)) "; magic or epoch differs" else "",
                });
                return k.fail(3);
            }
            if (k.warn_ns > 0 and wire.monotonicNs() -| t0 > k.warn_ns) k.logExchange(q, site, bytes, t0, "done (this rank's coll.collect waited for it)");
            @atomicStore(u64, &k.shared.slot[slot].done, q, .release);
            _ = k.exchanged.fetchAdd(1, .monotonic);
            q += 1;
        }
        return 0;
    }

    /// Both completions of gather q: the local send (TX[s] may be re-staged) and the peer's write (RX[s] is complete).
    fn completions(k: *Core, q: u64, site: u32, bytes: u64, len: usize, t0: u64, wc: *[1]vabi.Wc) u32 {
        var have_send = false;
        var have_recv = false;
        var last = t0;
        var spins: u32 = 0;
        while (!(have_send and have_recv)) {
            if (!have_send) {
                const n = k.nic.poll(k.nic.cq_send, wc);
                if (n < 0) return k.fail(5);
                for (wc[0..@intCast(n)]) |w| {
                    if (w.status != vabi.wc_success) {
                        k.logExchange(q, site, bytes, t0, k.nic.statusText(w.status));
                        return k.fail(nic.statusCode(w.status));
                    }
                    k.sends += 1;
                    if (w.wr_id == q) have_send = true;
                }
                if (n > 0) last = wire.monotonicNs();
            }
            if (!have_recv) {
                const n = k.nic.poll(k.nic.cq_recv, wc);
                if (n < 0) return k.fail(5);
                for (wc[0..@intCast(n)]) |w| {
                    if (w.status != vabi.wc_success) {
                        k.logExchange(q, site, bytes, t0, k.nic.statusText(w.status));
                        return k.fail(nic.statusCode(w.status));
                    }
                    k.recvs += 1;
                    k.nic.postRecv() catch {
                        k.logExchange(q, site, bytes, t0, "ibv_post_recv failed");
                        return k.fail(5);
                    };
                    if (w.opcode != nic.wc_recv_rdma_with_imm or (w.wc_flags & vabi.wc_with_imm) == 0 or w.imm_data != nic.immediate(q) or w.byte_len != len) {
                        std.log.warn("collectives: rank {d} proxy: the peer's write for gather {d} ({s} layer {d}, {d} bytes) carries imm {d} and {d} bytes (expected {d} and {d})", .{
                            k.rank, q, tcp_proxy.phaseName(site), site & 0xffff, bytes, std.mem.bigToNative(u32, w.imm_data), w.byte_len, @as(u32, @truncate(q)), len,
                        });
                        return k.fail(3);
                    }
                    have_recv = true;
                }
                if (n > 0) last = wire.monotonicNs();
            }
            if (have_send and have_recv) break;
            if (k.stop.load(.acquire) != 0) return 0;
            if (@atomicLoad(u32, &k.shared.abort, .acquire) != 0) {
                tcp_proxy.deviceGaveUp(k.shared, k.rank, q, k.nslot, "waiting for the completions of gather");
                return k.fail(2);
            }
            spins +%= 1;
            if (spins % 256 == 0) {
                const now = wire.monotonicNs();
                if (now -| last > k.progress_ns) {
                    k.logExchange(q, site, bytes, t0, if (have_send) "no write from the peer within the progress deadline (Timeout)" else "no send completion within the progress deadline (Timeout)");
                    return k.fail(2);
                }
                if (k.peer_fd >= 0 and spins % 4096 == 0 and peerGone(k.peer_fd)) {
                    k.logExchange(q, site, bytes, t0, "the peer closed its control socket (PeerLost)");
                    return k.fail(1);
                }
            }
            std.atomic.spinLoopHint();
        }
        return 0;
    }

    fn fail(k: *Core, code: u32) u32 {
        tcp_proxy.raiseAbort(k.shared);
        return code;
    }

    fn logExchange(k: *const Core, q: u64, site: u32, bytes: u64, t0: u64, outcome: []const u8) void {
        std.log.warn("collectives: rank {d} rdma proxy exchange of gather {d} ({s} layer {d}, {d} bytes) after {d} ms: {s}", .{
            k.rank, q, tcp_proxy.phaseName(site), site & 0xffff, bytes, (wire.monotonicNs() -| t0) / std.time.ns_per_ms, outcome,
        });
    }
};

/// True when the peer's end of `fd` is closed (a read would return end of stream); data or nothing to read: alive.
fn peerGone(fd: c.fd_t) bool {
    var b: [1]u8 = undefined;
    const r = c.recv(fd, &b, 1, c.MSG.PEEK | c.MSG.DONTWAIT);
    if (r > 0) return false;
    if (r == 0) return true;
    return switch (c.errno(r)) {
        .AGAIN, .INTR => false,
        else => true,
    };
}
