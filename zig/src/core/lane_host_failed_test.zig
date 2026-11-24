//! A lane host's status says failed once its family's hook does (the server's /health then answers 503).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

var broke: bool = false;

fn pairBroke() bool {
    return broke;
}

fn statusOf(hook: ?*const fn () bool) !api.Status {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    host.failed = hook;
    var status: api.Status = .{};
    host.engine().status(&status, &.{});
    return status;
}

test "a lane host's status is failed only while its family's hook says so" {
    broke = false;
    try std.testing.expect(!(try statusOf(null)).failed);
    try std.testing.expect(!(try statusOf(pairBroke)).failed);
    broke = true;
    try std.testing.expect((try statusOf(pairBroke)).failed);
    try std.testing.expect(!(try statusOf(null)).failed);
}
