//! prompt_cache's transfers between engines (POST /cache/export and /cache/import), on the Fake family of its tests.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Store = pc.Store;
const tests = @import("prompt_cache_test.zig");
const Fake = tests.Fake;
const fresh = tests.fresh;

test "a kept state leaves one store as bytes and resumes in another of the same identity, equal to a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
    const thread = [_]u32{ 4, 8, 15, 16, 23, 42, 7, 9 };
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    {
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.moving(), rules, 1 << 20);
        defer s.deinit();
        s.identity = 77;
        _ = f.pass(&s, &thread, try s.begin(a, &thread, 6, &.{}, &.{}, null));
        try std.testing.expectEqual(@as(u32, 6), s.probe(&thread, &.{}));
        try std.testing.expectEqual(@as(?u32, 6), try s.exportState(gpa, &thread, &.{}, &bytes));
        var none: std.ArrayList(u8) = .empty;
        defer none.deinit(gpa);
        try std.testing.expectEqual(@as(?u32, null), try s.exportState(gpa, &[_]u32{ 1, 2, 3 }, &.{}, &none)); // nothing kept covers it
    }
    var g: Fake = .{ .gpa = gpa };
    var t = Store.init(gpa, g.moving(), rules, 1 << 20);
    defer t.deinit();
    t.identity = 77;
    try std.testing.expectEqual(@as(u32, 0), t.probe(&thread, &.{}));
    try std.testing.expectEqual(@as(u32, 6), try t.importState(bytes.items));
    try std.testing.expectEqual(@as(u32, 6), try t.importState(bytes.items)); // the same state twice: kept once
    try std.testing.expectEqual(@as(usize, 1), t.entries.items.len);
    try std.testing.expectEqual(@as(u32, 6), t.probe(&thread, &.{}));
    const next = thread ++ [_]u32{ 3, 1 }; // the thread's next turn
    const plan = try t.begin(a, &next, 8, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 6), plan.from);
    try std.testing.expectEqual(fresh(&next), g.pass(&t, &next, plan));
    var h: Fake = .{ .gpa = gpa };
    var u = Store.init(gpa, h.moving(), rules, 1 << 20);
    defer u.deinit();
    u.identity = 78; // another build
    try std.testing.expectError(error.OtherIdentity, u.importState(bytes.items));
    u.identity = 77;
    try std.testing.expectError(error.BadState, u.importState(bytes.items[0 .. bytes.items.len - 1]));
    bytes.items[0] ^= 1;
    try std.testing.expectError(error.BadState, u.importState(bytes.items));
}
