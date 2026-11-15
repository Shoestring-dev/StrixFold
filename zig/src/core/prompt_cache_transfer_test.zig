//! prompt_cache's transfers (POST /cache/export, /cache/import) through the Store's public API, on a fake family here.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Store = pc.Store;
const Saved = pc.Saved;
const Allocator = std.mem.Allocator;

/// A family over host memory: its live state is a position and a running sum; a block is one byte a position.
const Fake = struct {
    gpa: Allocator,
    at: u32 = 0,
    sum: u64 = 0,
    live: usize = 0,
    fail_import: bool = false,

    const State = struct { at: u32, sum: u64 };

    fn moving(f: *Fake) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .export_state = exportFn, .import_state = importFn } };
    }
    fn still(f: *Fake) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn } };
    }
    fn of(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytesFn(_: *anyopaque, at: u32) u64 {
        return 100 + at;
    }
    fn saveFn(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!Saved {
        const f = of(ptr);
        if (at != f.at) return error.NotAtMark;
        const st = try f.gpa.create(State);
        st.* = .{ .at = f.at, .sum = f.sum };
        f.live += 1;
        return st;
    }
    fn restoreFn(ptr: *anyopaque, _: ?*anyopaque, saved: Saved) anyerror!void {
        const st: *State = @ptrCast(@alignCast(saved));
        of(ptr).at, of(ptr).sum = .{ st.at, st.sum };
    }
    fn dropFn(ptr: *anyopaque, saved: Saved) void {
        of(ptr).gpa.destroy(@as(*State, @ptrCast(@alignCast(saved))));
        of(ptr).live -= 1;
    }
    /// The slot: the position and the sum (12 bytes); the rows: one byte a position, the position's low byte.
    fn exportFn(_: *anyopaque, saved: Saved, gpa: Allocator, slot: *[]u8, rows: *[]u8) anyerror!void {
        const st: *State = @ptrCast(@alignCast(saved));
        slot.* = try gpa.alloc(u8, 12);
        errdefer gpa.free(slot.*);
        std.mem.writeInt(u32, slot.*[0..4], st.at, .little);
        std.mem.writeInt(u64, slot.*[4..12], st.sum, .little);
        rows.* = try gpa.alloc(u8, st.at);
        for (rows.*, 0..) |*b, i| b.* = @truncate(i);
    }
    fn importFn(ptr: *anyopaque, at: u32, slot: []const u8, rows: []const u8) anyerror!Saved {
        const f = of(ptr);
        if (f.fail_import) return error.CopyFailed;
        if (slot.len != 12 or rows.len != at or std.mem.readInt(u32, slot[0..4], .little) != at) return error.BadState;
        for (rows, 0..) |b, i| if (b != @as(u8, @truncate(i))) return error.BadState;
        const st = try f.gpa.create(State);
        st.* = .{ .at = at, .sum = std.mem.readInt(u64, slot[4..12], .little) };
        f.live += 1;
        return st;
    }
    /// A prompt pass from `plan.from`, keeping at each mark; the sum a fresh pass would give.
    fn pass(f: *Fake, s: *Store, prompt: []const u32, plan: pc.Plan) u64 {
        if (plan.from == 0) f.* = .{ .gpa = f.gpa, .live = f.live };
        var mi: usize = 0;
        for (prompt[plan.from..]) |t| {
            f.sum = f.sum *% 31 +% t;
            f.at += 1;
            if (mi < plan.marks.len and plan.marks[mi] == f.at) {
                _ = s.keep(prompt, f.at, null, &.{});
                mi += 1;
            }
        }
        return f.sum;
    }
};

fn fresh(prompt: []const u32) u64 {
    var sum: u64 = 0;
    for (prompt) |t| sum = sum *% 31 +% t;
    return sum;
}

const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
const thread = [_]u32{ 4, 8, 15, 16, 23, 42, 7, 9 };

/// A store that kept `thread`'s state at 6, exported: the caller frees the result's ids and parts.
fn exported(gpa: Allocator, a: Allocator, f: *Fake) !pc.Exported {
    var s = Store.init(gpa, f.moving(), rules, 1 << 20);
    defer s.deinit();
    _ = f.pass(&s, &thread, try s.begin(a, &thread, 6, &.{}, &.{}, null));
    try std.testing.expectEqual(@as(u32, 6), s.probe(&thread, &.{}));
    try std.testing.expectEqual(@as(u32, 0), s.probe(&.{ 1, 2, 3 }, &.{}));
    try std.testing.expect(try s.exportState(gpa, &.{ 1, 2, 3 }, &.{}) == null); // nothing kept covers it
    return (try s.exportState(gpa, &thread, &.{})).?;
}

test "a kept state leaves one store and resumes in another, equal to a fresh pass; the same bytes twice keep once" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    const x = try exported(gpa, a, &f);
    defer gpa.free(x.ids);
    defer gpa.free(x.parts.slot);
    defer gpa.free(x.parts.rows);
    try std.testing.expectEqual(@as(u32, 6), x.at);
    try std.testing.expectEqualSlices(u32, thread[0..7], x.ids); // the positions and the lookahead token
    try std.testing.expectEqual(@as(usize, 6), x.parts.rows.len);
    var g: Fake = .{ .gpa = gpa };
    var t = Store.init(gpa, g.moving(), rules, 1 << 20);
    defer t.deinit();
    try std.testing.expectEqual(@as(u32, 0), t.probe(&thread, &.{}));
    try std.testing.expectEqual(@as(u32, 6), try t.importState(x.ids, x.at, x.parts.slot, x.parts.rows));
    try std.testing.expectEqual(@as(u32, 6), try t.importState(x.ids, x.at, x.parts.slot, x.parts.rows));
    try std.testing.expectEqual(@as(usize, 1), t.entries.items.len);
    try std.testing.expectEqual(@as(u64, 1), t.counts.kept);
    try std.testing.expectEqual(@as(u32, 6), t.probe(&thread, &.{}));
    const next = thread ++ [_]u32{ 3, 1 }; // the thread's next turn
    const plan = try t.begin(a, &next, 8, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 6), plan.from);
    try std.testing.expectEqual(fresh(&next), g.pass(&t, &next, plan));
}

test "an import is refused whole: a family without transfers, ids off the positions, the family's refusal, no room" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    const x = try exported(gpa, a, &f);
    defer gpa.free(x.ids);
    defer gpa.free(x.parts.slot);
    defer gpa.free(x.parts.rows);
    var g: Fake = .{ .gpa = gpa };
    var fixed = Store.init(gpa, g.still(), rules, 1 << 20);
    defer fixed.deinit();
    try std.testing.expectError(error.NoTransfer, fixed.importState(x.ids, x.at, x.parts.slot, x.parts.rows));
    try std.testing.expectError(error.NoTransfer, fixed.exportState(gpa, &thread, &.{}));
    var t = Store.init(gpa, g.moving(), rules, 1 << 20);
    defer t.deinit();
    try std.testing.expectError(error.BadState, t.importState(x.ids[0..6], x.at, x.parts.slot, x.parts.rows));
    try std.testing.expectError(error.BadState, t.importState(x.ids, 0, x.parts.slot, x.parts.rows));
    try std.testing.expectError(error.BadState, t.importState(x.ids, x.at, x.parts.slot[0..11], x.parts.rows));
    g.fail_import = true;
    try std.testing.expectError(error.CopyFailed, t.importState(x.ids, x.at, x.parts.slot, x.parts.rows));
    g.fail_import = false;
    try std.testing.expectEqual(@as(usize, 0), t.entries.items.len);
    try std.testing.expectEqual(@as(usize, 0), g.live);
    var small = Store.init(gpa, g.moving(), rules, 50); // the budget holds no state of 106 bytes
    defer small.deinit();
    try std.testing.expectError(error.PastBudget, small.importState(x.ids, x.at, x.parts.slot, x.parts.rows));
    // the budget holds one: an import evicts the older kept state for the new one
    var one = Store.init(gpa, g.moving(), rules, 150);
    defer one.deinit();
    const other = [_]u32{ 9, 9, 9, 9, 9, 9, 9, 9 };
    _ = g.pass(&one, &other, try one.begin(a, &other, 6, &.{}, &.{}, null));
    try std.testing.expectEqual(@as(u32, 6), one.probe(&other, &.{}));
    try std.testing.expectEqual(@as(u32, 6), try one.importState(x.ids, x.at, x.parts.slot, x.parts.rows));
    try std.testing.expectEqual(@as(u32, 0), one.probe(&other, &.{}));
    try std.testing.expectEqual(@as(u64, 1), one.counts.evicted);
}
