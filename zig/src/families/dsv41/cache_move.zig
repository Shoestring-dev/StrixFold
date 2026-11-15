//! POST /cache/export and /cache/import for dsv41: a kept state in sf-disk's layout off the device and back, each rank.
const std = @import("std");
const builtin = @import("builtin");
const gpu = @import("gpu");
const core = @import("core");
const model = @import("model.zig");
const disk = @import("disk.zig");
const identity_mod = @import("disk_identity.zig");
const lanes_mod = @import("lanes.zig");
const link = @import("link.zig");
const pd = core.prompt_disk;
const Model = model.Model;
const Lanes = lanes_mod.Lanes;
const Snap = lanes_mod.Snap;
const Reader = lanes_mod.Reader;
const block_tokens = disk.block_tokens;

/// What this rank's states look like on the wire (native/hip.zig hands it to the prompt cache as cache_wire.Shape).
pub const Shape = struct {
    ns: [16]u8, // sf-disk's namespace of this rank's identity text
    block_tokens: u32,
    max_blocks: u32,
    max_positions: u32,
    slot_bytes: u64,
    block_bytes: u64,
};

/// A cache_put frame's bytes at most (a follower's frame buffer holds a full pool's ids and 64 KiB).
pub const piece = lanes_mod.image_piece;

/// The shape (HIP, as sf-disk's identity), or null with one log line: the launch has no identity, so no state moves.
pub fn shapeOf(gpa: std.mem.Allocator, m: *const Model) ?Shape {
    if (comptime gpu.backend != .hip) return null;
    return shapeOr(gpa, m) catch |e| {
        note("cache transfers off: {s}", .{@errorName(e)});
        return null;
    };
}

fn shapeOr(gpa: std.mem.Allocator, m: *const Model) !Shape {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const planes = try disk.planesOf(a, m);
    const parts = try identity_mod.identity(a, m, planes);
    const id = try identity_mod.identityText(a, parts);
    return .{ .ns = pd.namespaceOf(id), .block_tokens = block_tokens, .max_blocks = pd.max_blocks, .max_positions = @intCast(m.context_cap), .slot_bytes = m.snapBytes(), .block_bytes = blockBytes(planes) };
}

/// One block's bytes, every plane.
fn blockBytes(planes: []const disk.Plane) u64 {
    var n: u64 = 0;
    for (planes) |p| n += disk.blockBytes(p);
    return n;
}

// -- rank 0 (the prompt cache's Snapshots functions) ------------------------------------------------------------

/// POST /cache/export: the state's slot bytes and its blocks' rows (block by block, plane by plane) off the device.
pub fn exportFn(ptr: *anyopaque, saved: *anyopaque, gpa: std.mem.Allocator, slot_out: *[]u8, rows_out: *[]u8) anyerror!void {
    const self: *Lanes = @ptrCast(@alignCast(ptr));
    const sn: *Snap = @ptrCast(@alignCast(saved));
    if (sn.stale or sn.at == 0 or sn.at % block_tokens != 0) return error.StaleState;
    const m = self.m;
    const planes = try disk.planesOf(gpa, m);
    defer gpa.free(planes);
    try m.stream.synchronize(); // (the copies that made the state are on the stream)
    const slot = try gpa.alloc(u8, m.snapBytes());
    errdefer gpa.free(slot);
    try gpu.copyD2H(m.ctx.d, slot, sn.buf.ptr);
    const k = sn.at / block_tokens;
    const rows = try gpa.alloc(u8, k * blockBytes(planes));
    errdefer gpa.free(rows);
    var off: usize = 0;
    for (1..k + 1) |j| for (planes) |p| {
        const n: usize = @intCast(disk.blockBytes(p));
        try gpu.copyD2H(m.ctx.d, rows[off..][0..n], disk.rowsAt(p, sn.base, j));
        off += n;
    };
    slot_out.* = slot;
    rows_out.* = rows;
}

/// POST /cache/import: an exported state as a kept one here and on rank 1 (cache_put pieces): both ranks or neither.
pub fn importFn(ptr: *anyopaque, at: u32, slot: []const u8, rows: []const u8) anyerror!*anyopaque {
    const self: *Lanes = @ptrCast(@alignCast(ptr));
    return importState(self, at, slot, rows) catch |err| {
        lanes_mod.ended(err);
        return err;
    };
}

fn importState(self: *Lanes, at32: u32, slot: []const u8, rows: []const u8) !*anyopaque {
    const m = self.m;
    const gpa = self.gpa;
    const at: usize = at32;
    if (at == 0 or at % block_tokens != 0 or at > m.context_cap) return error.NotAChunkEnd;
    if (slot.len != m.snapBytes()) return error.BadState;
    const planes = try disk.planesOf(gpa, m);
    defer gpa.free(planes);
    const k = at / block_tokens;
    if (rows.len != k * blockBytes(planes)) return error.BadRows;
    var taken: [model.max_streams][2]usize = undefined;
    var nt: usize = 0;
    var it = self.streams.valueIterator();
    while (it.next()) |l| {
        taken[nt] = .{ l.base, l.end };
        nt += 1;
    }
    const base = try self.placeKept(taken[0..nt], at, null);
    self.staleOver(base, base + at, null);
    try m.stream.synchronize();
    const sn = try gpa.create(Snap);
    errdefer gpa.destroy(sn);
    sn.* = .{ .id = self.next_snap, .at = at, .base = base, .end = base + at, .buf = try gpu.DeviceBuffer.alloc(m.ctx.d, slot.len) };
    errdefer sn.buf.free();
    self.next_snap += 1;
    try gpu.copyH2D(m.ctx.d, sn.buf.ptr, slot);
    var off: usize = 0;
    for (1..k + 1) |j| for (planes) |p| {
        const n: usize = @intCast(disk.blockBytes(p));
        try gpu.copyH2D(m.ctx.d, disk.rowsAt(p, base, j), rows[off..][0..n]);
        off += n;
    };
    if (self.peer) |peer| {
        try sendPieces(self, sn.id, base, at, slot, rows);
        // rank 1 answers how its copies went (0 ok, 1 bad); rank 0 tells it whether the state stands
        var rb: [16]u8 = undefined;
        const got = try peer.recv(&rb);
        if (got.len != 1 or got[0] > 1) return error.BadFrame;
        const ok = got[0] == 0;
        if (try self.begin(.cache_outcome)) {
            try self.wire.int(gpa, u64, sn.id);
            try self.wire.int(gpa, u8, @intFromBool(ok));
            try self.flush();
        }
        if (!ok) return error.PeerImportFailed;
    }
    self.snaps.append(gpa, sn) catch |err| {
        self.sendDrop(sn.id) catch {};
        return err;
    };
    return sn;
}

/// The state's bytes to rank 1 in cache_put frames: the id, extent, positions, total and offset, then the piece.
fn sendPieces(self: *Lanes, id: u64, base: usize, at: usize, slot: []const u8, rows: []const u8) !void {
    const gpa = self.gpa;
    const total = slot.len + rows.len;
    var off: usize = 0;
    while (off < total) {
        const n = @min(piece, total - off);
        if (try self.begin(.cache_put)) {
            try self.wire.int(gpa, u64, id);
            try self.wire.int(gpa, u64, base);
            try self.wire.int(gpa, u64, at);
            try self.wire.int(gpa, u64, total);
            try self.wire.int(gpa, u64, off);
            try appendSpan(gpa, &self.wire.buf, slot, rows, off, n);
            try self.flush();
        }
        off += n;
    }
}

/// Bytes [off, off + n) of the state as the body orders them (the slot, then the rows), appended to `buf`.
fn appendSpan(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), slot: []const u8, rows: []const u8, off: usize, n: usize) !void {
    const from_slot = if (off < slot.len) @min(n, slot.len - off) else 0;
    if (from_slot > 0) try buf.appendSlice(gpa, slot[off..][0..from_slot]);
    if (n > from_slot) try buf.appendSlice(gpa, rows[off + from_slot - slot.len ..][0 .. n - from_slot]);
}

// -- rank 1 (lanes.follow) ---------------------------------------------------------------------------------------

/// A follower's side of an import: the pieces as they come, its copies after the last, kept once rank 0 says so.
pub const Taker = struct {
    put: std.ArrayList(u8) = .empty,
    held: ?struct { id: u64, buf: gpu.DeviceBuffer } = null,

    pub fn deinit(t: *Taker, gpa: std.mem.Allocator) void {
        t.put.deinit(gpa);
        if (t.held) |h| {
            var b = h.buf;
            b.free();
        }
        t.held = null;
    }

    /// A cache_put frame: offset and length checked against the total (the total against this rank's sizes first).
    pub fn piece(t: *Taker, gpa: std.mem.Allocator, m: *Model, r: *Reader, peer: link.Link) !void {
        const id = try r.int(u64);
        const base: usize = @intCast(try r.int(u64));
        const at: usize = @intCast(try r.int(u64));
        const total: usize = @intCast(try r.int(u64));
        const off: usize = @intCast(try r.int(u64));
        const p = r.rest();
        try r.end();
        if (off != t.put.items.len or p.len == 0 or off + p.len > total) return error.BadFrame;
        if (off == 0) {
            if (at == 0 or at % block_tokens != 0 or at > m.pool_cap or base + at > m.pool_cap) return error.BadFrame;
            if (total != m.snapBytes() + try rowsBytes(gpa, m, at)) return error.BadFrame;
        }
        try t.put.appendSlice(gpa, p);
        if (t.put.items.len < total) return;
        defer t.put.clearRetainingCapacity();
        const ok = t.take(gpa, m, id, base, at);
        if (!ok) std.log.warn("lanes: rank 1 could not take imported state {d} ({d} positions)", .{ id, at });
        try peer.send(&.{@intFromBool(!ok)});
    }

    fn take(t: *Taker, gpa: std.mem.Allocator, m: *Model, id: u64, base: usize, at: usize) bool {
        if (t.held) |h| { // (an earlier import's outcome never came: rank 0 forgot it)
            var b = h.buf;
            b.free();
            t.held = null;
        }
        return t.takeOr(gpa, m, id, base, at) catch false;
    }

    fn takeOr(t: *Taker, gpa: std.mem.Allocator, m: *Model, id: u64, base: usize, at: usize) !bool {
        const planes = try disk.planesOf(gpa, m);
        defer gpa.free(planes);
        const sb = m.snapBytes();
        try m.stream.synchronize();
        var b = try gpu.DeviceBuffer.alloc(m.ctx.d, sb);
        errdefer b.free();
        try gpu.copyH2D(m.ctx.d, b.ptr, t.put.items[0..sb]);
        var off: usize = sb;
        for (1..at / block_tokens + 1) |j| for (planes) |p| {
            const n: usize = @intCast(disk.blockBytes(p));
            try gpu.copyH2D(m.ctx.d, disk.rowsAt(p, base, j), t.put.items[off..][0..n]);
            off += n;
        };
        t.held = .{ .id = id, .buf = b };
        return true;
    }

    /// A cache_outcome frame: the state taken goes into `kept` under its id, or is freed.
    pub fn outcome(t: *Taker, gpa: std.mem.Allocator, r: *Reader, kept: *std.AutoHashMapUnmanaged(u64, gpu.DeviceBuffer)) !void {
        const id = try r.int(u64);
        const keep = try r.int(u8);
        try r.end();
        if (keep > 1) return error.BadFrame;
        const h = t.held orelse return; // (nothing taken: rank 0 heard so and drops its own)
        t.held = null;
        var b = h.buf;
        if (h.id != id or keep == 0) return b.free();
        kept.put(gpa, id, b) catch b.free();
    }
};

/// The rows a state of `at` positions holds on this rank, every block and plane.
fn rowsBytes(gpa: std.mem.Allocator, m: *const Model, at: usize) !u64 {
    const planes = try disk.planesOf(gpa, m);
    defer gpa.free(planes);
    return blockBytes(planes) * (at / block_tokens);
}

fn note(comptime fmt: []const u8, args: anytype) void {
    if (builtin.is_test) return;
    std.log.info("dsv41: " ++ fmt, args);
}

test "a cache_put frame is refused on its offset, an empty piece, or bytes past its total, before any is kept" {
    const gpa = std.testing.allocator;
    var t: Taker = .{};
    defer t.deinit(gpa);
    try t.put.appendSlice(gpa, &.{ 1, 2, 3 }); // three bytes already in: the next piece starts at 3
    var w: lanes_mod.Wire = .{};
    defer w.deinit(gpa);
    const Case = struct { off: u64, total: u64, bytes: []const u8 };
    for ([_]Case{ .{ .off = 0, .total = 10, .bytes = "ab" }, .{ .off = 3, .total = 10, .bytes = "" }, .{ .off = 3, .total = 4, .bytes = "ab" } }) |c| {
        try w.begin(gpa, .cache_put, 7);
        try w.int(gpa, u64, 1);
        try w.int(gpa, u64, 0);
        try w.int(gpa, u64, 2048);
        try w.int(gpa, u64, c.total);
        try w.int(gpa, u64, c.off);
        try w.buf.appendSlice(gpa, c.bytes);
        var r: Reader = .{ .b = w.buf.items[9..] };
        const m = try gpa.create(Model); // (never read: the frame's shape refuses it first)
        defer gpa.destroy(m);
        try std.testing.expectError(error.BadFrame, t.piece(gpa, m, &r, .{ .fd = -1 }));
        try std.testing.expectEqual(@as(usize, 3), t.put.items.len);
    }
    // an outcome for nothing taken is a no-op; a keep byte other than 0 or 1 is refused
    var kept: std.AutoHashMapUnmanaged(u64, gpu.DeviceBuffer) = .empty;
    defer kept.deinit(gpa);
    try w.begin(gpa, .cache_outcome, 8);
    try w.int(gpa, u64, 1);
    try w.int(gpa, u8, 1);
    var r: Reader = .{ .b = w.buf.items[9..] };
    try t.outcome(gpa, &r, &kept);
    try std.testing.expectEqual(@as(usize, 0), kept.count());
    try w.begin(gpa, .cache_outcome, 9);
    try w.int(gpa, u64, 1);
    try w.int(gpa, u8, 2);
    var r2: Reader = .{ .b = w.buf.items[9..] };
    try std.testing.expectError(error.BadFrame, t.outcome(gpa, &r2, &kept));
}

test "an import's pieces cover the slot then the rows, in order and whole, at any piece size" {
    const gpa = std.testing.allocator;
    const slot = "abcdefg";
    const rows = "0123456789";
    for ([_]usize{ 1, 4, 7, 8, 17, 100 }) |n_piece| {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        var off: usize = 0;
        while (off < slot.len + rows.len) {
            const n = @min(n_piece, slot.len + rows.len - off);
            try appendSpan(gpa, &out, slot, rows, off, n);
            off += n;
        }
        try std.testing.expectEqualStrings(slot ++ rows, out.items);
    }
}
