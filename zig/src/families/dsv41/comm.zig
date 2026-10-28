//! The TP ranks' collectives: NCCL opened at run time, its unique id passed from rank 0 over the link. Every model
//! reduction is an all-gather of fp32 partials that each rank then adds in rank order (DESIGN.md section 3), so the
//! ranks hold the same bits without a broadcast.
const std = @import("std");
const gpu = @import("gpu");
const Link = @import("link.zig").Link;
const Fan = @import("fan.zig").Fan;
const coll_mod = @import("coll.zig");

const nccl = gpu.nccl;

pub const Comm = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: u32,
    world: u32,
    /// The Coll's first failure (coll.zig: it poisons every later call): 0 none, else a Poison.
    poison: std.atomic.Value(u8) = .init(0),

    const Poison = enum(u8) { none, failed, aborted };

    /// Rank 0 makes the id and sends it; the others take it from the link. Needs the CUDA context current.
    pub fn init(link: Link, rank: u32, world: u32) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(std.mem.asBytes(&id));
        } else {
            var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
            const got = try link.recv(&buf);
            if (got.len != buf.len) return error.BadUniqueId;
            id = std.mem.bytesToValue(nccl.UniqueId, got);
        }
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    /// Comm over every link of a fan: rank 0 makes the id and sends it to each follower. Needs the CUDA context current.
    pub fn fromFan(f: *const Fan, rank: u32, world: u32) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (world > 1) {
            if (rank == 0) {
                try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
                for (f.links) |l| if (l) |x| try x.send(std.mem.asBytes(&id));
            } else {
                var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
                const got = try f.links[0].?.recv(&buf);
                if (got.len != buf.len) return error.BadUniqueId;
                id = std.mem.bytesToValue(nccl.UniqueId, got);
            }
        } else try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        var c: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&c, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = c, .rank = rank, .world = world };
    }

    /// This communicator as the family's collectives (coll.zig): bytes moved as u8, which is what an all-gather or an
    /// exchange of any type is. NCCL's calls are graph-capturable; NCCL carries no header, so the site is not checked
    /// against the peer's. `c` must outlive the Coll.
    pub fn coll(c: *Comm) coll_mod.Coll {
        return .{ .ptr = c, .vtable = &coll_vtable, .rank = c.rank, .world = c.world, .capturable = true };
    }

    const coll_vtable: coll_mod.Coll.VTable = .{ .all_gather = collGather, .exchange = collExchange, .abort = collAbort, .close = collClose };

    fn poisoned(c: *Comm) coll_mod.Error!void {
        return switch (@as(Poison, @enumFromInt(c.poison.load(.acquire)))) {
            .none => {},
            .failed => error.CollectiveFailed,
            .aborted => error.Aborted,
        };
    }

    fn fail(c: *Comm) coll_mod.Error {
        _ = c.poison.cmpxchgStrong(@intFromEnum(Poison.none), @intFromEnum(Poison.failed), .acq_rel, .acquire);
        return if (c.poisoned()) |_| error.CollectiveFailed else |e| e;
    }

    fn collGather(ptr: *anyopaque, stream: gpu.Stream, _: coll_mod.Site, send: u64, recv: u64, bytes: usize) coll_mod.Error!void {
        const c: *Comm = @ptrCast(@alignCast(ptr));
        try c.poisoned();
        c.allGather(send, recv, bytes, .u8, stream) catch return c.fail();
    }

    fn collExchange(ptr: *anyopaque, stream: gpu.Stream, _: coll_mod.Site, send: u64, send_bytes: usize, recv: u64, recv_bytes: usize, peer: u32) coll_mod.Error!void {
        const c: *Comm = @ptrCast(@alignCast(ptr));
        try c.poisoned();
        c.exchange(send, send_bytes, recv, recv_bytes, .u8, peer, stream) catch return c.fail();
    }

    /// Every later call fails with Aborted. NCCL's abort is not in the runtime's table, so a kernel already enqueued is
    /// not interrupted: the caller stops issuing and closes.
    fn collAbort(ptr: *anyopaque, reason: coll_mod.Reason) void {
        const c: *Comm = @ptrCast(@alignCast(ptr));
        if (c.poison.cmpxchgStrong(@intFromEnum(Poison.none), @intFromEnum(Poison.aborted), .acq_rel, .acquire) == null)
            std.log.warn("NCCL collectives aborted ({s}); closing", .{@tagName(reason)});
    }

    fn collClose(ptr: *anyopaque) void {
        const c: *Comm = @ptrCast(@alignCast(ptr));
        c.deinit();
    }

    pub fn deinit(c: *Comm) void {
        _ = c.lib.api.ncclCommDestroy(c.comm);
        c.lib.close();
        c.* = undefined;
    }

    /// `count` elements of `dtype` from every rank into `recv`, rank after rank.
    pub fn allGather(c: Comm, send: u64, recv: u64, count: usize, dtype: nccl.DataType, stream: gpu.Stream) !void {
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, dtype, c.comm, stream.handle), "ncclAllGather");
    }

    /// `count` elements of `dtype` to rank `peer` and `rcount` from it, in one NCCL group (point to point).
    pub fn exchange(c: Comm, send: u64, count: usize, recv: u64, rcount: usize, dtype: nccl.DataType, peer: u32, stream: gpu.Stream) !void {
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        const s = c.lib.api.ncclSend(send, count, dtype, @intCast(peer), c.comm, stream.handle);
        const r = c.lib.api.ncclRecv(recv, rcount, dtype, @intCast(peer), c.comm, stream.handle);
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        try c.lib.check(s, "ncclSend");
        try c.lib.check(r, "ncclRecv");
    }
};
