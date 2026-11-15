//! LaneHost's prompt-cache transfers: HTTP threads queue them, the engine's thread runs them between rounds.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

fn self(ctx: *anyopaque) *LaneHost {
    return @ptrCast(@alignCast(ctx));
}

/// The store's wire shape (set once at open, so read without the lock); null without a store or a shape.
pub fn shapeFn(ctx: *anyopaque) ?api.CacheShape {
    const store = self(ctx).cache orelse return null;
    return store.shape;
}

/// A transfer queued for the engine thread, which owns the store and the family, and waited for.
pub fn cacheFn(ctx: *anyopaque, op: *api.CacheOp) void {
    const h = self(ctx);
    h.mutex.lockUncancelable(h.io);
    defer h.mutex.unlock(h.io);
    if (h.cache == null or h.closing) {
        op.err = if (h.closing) error.Closed else error.NoCache;
        op.done = true;
        return;
    }
    h.ops.append(h.gpa, op) catch {
        op.err = error.OutOfMemory;
        op.done = true;
        return;
    };
    h.wake.signal(h.io);
    while (!op.done) h.ops_done.waitUncancelable(h.io, &h.mutex);
}

/// The queued transfers, run here between rounds; each caller is woken when its own is done.
pub fn takeOps(h: *LaneHost) void {
    h.mutex.lockUncancelable(h.io);
    if (h.ops.items.len == 0) return h.mutex.unlock(h.io);
    const ops = h.gpa.dupe(*api.CacheOp, h.ops.items) catch return h.mutex.unlock(h.io);
    h.ops.clearRetainingCapacity();
    h.mutex.unlock(h.io);
    defer h.gpa.free(ops);
    for (ops) |op| {
        runOp(h, op);
        finish(h, op);
    }
}

/// The host is closing: every queued transfer ends with Closed, its caller woken.
pub fn closeOps(h: *LaneHost) void {
    h.mutex.lockUncancelable(h.io);
    defer h.mutex.unlock(h.io);
    for (h.ops.items) |op| {
        op.err = error.Closed;
        op.done = true;
    }
    h.ops.clearRetainingCapacity();
    h.ops_done.broadcast(h.io);
}

fn finish(h: *LaneHost, op: *api.CacheOp) void {
    h.mutex.lockUncancelable(h.io);
    defer h.mutex.unlock(h.io);
    op.done = true;
    h.ops_done.broadcast(h.io);
}

fn runOp(h: *LaneHost, op: *api.CacheOp) void {
    const store = h.cache orelse {
        op.err = error.NoCache;
        return;
    };
    switch (op.kind) {
        .probe => op.tokens = store.probe(op.prompt, op.chunks),
        .export_state => {
            const x = store.exportState(op.gpa, op.prompt, op.chunks) catch |e| {
                op.err = e;
                return;
            };
            op.out = x orelse {
                op.err = error.NoState;
                return;
            };
            op.tokens = op.out.?.at;
        },
        .import_state => op.tokens = store.importState(op.prompt, op.at, op.slot, op.rows) catch |e| {
            op.err = e;
            return;
        },
    }
}

/// A family that keeps states in host memory for the test: bytes, save, restore, drop; no transfers.
const Still = struct {
    fn snapshots(s: *Still) pc.Snapshots {
        return .{ .ptr = s, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, _: u32) anyerror!pc.Saved {
        return ptr;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
    fn drop(_: *anyopaque, _: pc.Saved) void {}
};

test "a transfer runs on the engine thread and answers its caller; a closed host refuses it" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const e = host.engine();
    try std.testing.expect(e.cacheShape() == null);
    var op: api.CacheOp = .{ .kind = .probe, .prompt = &.{ 1, 2, 3 }, .gpa = gpa };
    try e.cacheOp(&op); // no store: refused at once, on the caller's thread
    try std.testing.expect(op.done and op.err.? == error.NoCache);
    var still: Still = .{};
    var store = pc.Store.init(gpa, still.snapshots(), .{}, 1 << 10);
    defer store.deinit();
    host.cache = &store;
    try host.start();
    op = .{ .kind = .probe, .prompt = &.{ 1, 2, 3 }, .gpa = gpa };
    try e.cacheOp(&op);
    try std.testing.expect(op.done and op.err == null);
    try std.testing.expectEqual(@as(u32, 0), op.tokens);
    op = .{ .kind = .export_state, .prompt = &.{ 1, 2, 3 }, .gpa = gpa };
    try e.cacheOp(&op);
    try std.testing.expect(op.done and op.err.? == error.NoTransfer);
    store.shape = .{ .ns = @splat(1), .block_tokens = 4, .max_blocks = 2, .max_positions = 8, .slot_bytes = 3, .block_bytes = 5 };
    try std.testing.expectEqual(@as(u64, 3), e.cacheShape().?.slot_bytes);
    host.stop();
    op = .{ .kind = .probe, .prompt = &.{ 1, 2, 3 }, .gpa = gpa };
    try e.cacheOp(&op);
    try std.testing.expect(op.done and op.err.? == error.Closed);
}
