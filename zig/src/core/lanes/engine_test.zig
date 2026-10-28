//! The round loop on the fake target: drafted == one-token rounds, shared rounds == solo, greedy and sampled.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;

const Case = struct {
    prompt: []const u32,
    max_new: u32 = 40,
    sampling: ?Sampling = null,
    drafts: bool = true,
    think_budget: u32 = 0,
};

fn model() !Config {
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
}

// The served HIP geometry: six rows/stream, sixteen per shared round, eight streams,
// five held proposals. This remains a host fake: it has no HIP tap rings or peer wire.
fn servedModel() !Config {
    const ms = [_]f64{ 36.06, 43.76, 50.38, 56.29, 59.52, 61.52, 66.27, 71.02, 75.77, 80.52, 85.27, 90.02, 94.77, 99.52, 104.27, 109.02 };
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, ms, 1..) |*c, value, width| c.* = .{ .width = @intCast(width), .ms = value };
    return Config.init(gpa, .{ .exact_width = 6, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 5, .window_costs = costs[0..6], .shared_costs = &costs, .mtp_step_ms = 0, .hidden_rows = true, .batch_rows = 16, .max_streams = 8, .draft_probabilities = true, .draft_streams = true }, 6, 5);
}

/// Every case's emitted tokens, the cases admitted together and stepped until done.
fn run(cases: []const Case) ![][]u32 {
    return runWithBypass(cases, null);
}

fn runWithBypass(cases: []const Case, bypass: ?u32) ![][]u32 {
    return runWithGeometry(cases, bypass, false);
}

fn runWithGeometry(cases: []const Case, bypass: ?u32, served: bool) ![][]u32 {
    var cfg = if (served) try servedModel() else try model();
    defer cfg.deinit(gpa);
    if (bypass) |at| {
        cfg.node_probabilities = true;
        try cfg.setDraftBypassAt(at);
    }
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |c, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91 });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

fn free(runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

const GuardCase = struct {
    prompt: []const u32,
    max_new: u32 = 400,
    drafts: bool = true,
    think_budget: u32 = 0,
    loop_guard: bool = true,
    think_open: ?bool = null,
    cycle_after: usize = 70,
    pattern: []const u32 = &.{ 11, 12, 13 },
    answer_cycles: bool = false,
};

const GuardOut = struct {
    emitted: []u32,
    reason: sm.Reason,
    period: ?u32,
    finished: bool,
};

fn runGuard(cases: []const GuardCase, step_limit: usize) ![]GuardOut {
    var cfg = try model();
    defer cfg.deinit(gpa);
    const c = cases[0];
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = c.cycle_after, .probe_prompt = c.prompt.len, .pattern = c.pattern, .answer_cycles = c.answer_cycles };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |item, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "guard", .prompt = item.prompt, .max_new = item.max_new, .eos = &.{96}, .drafts = item.drafts, .proposer = p.proposer(), .think_budget = item.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .think_open = item.think_open, .loop_guard = item.loop_guard });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    var steps: usize = 0;
    while (engine.activeCount() > 0 and steps < step_limit) : (steps += 1) try engine.step();
    const out = try gpa.alloc(GuardOut, cases.len);
    for (out, streams) |*result, *s| result.* = .{ .emitted = try gpa.dupe(u32, s.emitted()), .reason = s.reason, .period = s.loop_period, .finished = s.finished };
    return out;
}

fn freeGuard(runs: []GuardOut) void {
    for (runs) |run_result| gpa.free(run_result.emitted);
    gpa.free(runs);
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };

test "draft bypass preserves greedy and keyed sampled output with mixed streams and unequal endings" {
    for ([_]?Sampling{ null, .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 } }) |sampling| {
        const cases = [_]Case{
            .{ .prompt = &p1, .sampling = sampling, .max_new = 40 },
            .{ .prompt = &p2, .sampling = sampling, .max_new = 3 },
            .{ .prompt = &p1, .sampling = sampling, .max_new = 13, .drafts = false },
        };
        const normal = try runWithBypass(&cases, 0);
        defer free(normal);
        const bypassed = try runWithBypass(&cases, 2);
        defer free(bypassed);
        for (normal, bypassed) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    }
}

test "draft bypass clears held rows and resumes when active selected load falls" {
    var cfg = try model();
    defer cfg.deinit(gpa);
    cfg.node_probabilities = true;
    try cfg.setDraftBypassAt(2);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var one = try sm.Stream.init(gpa, .{ .id = "one", .prompt = &p1, .max_new = 40, .eos = &.{}, .drafts = true });
    defer one.deinit(gpa);
    var two = try sm.Stream.init(gpa, .{ .id = "two", .prompt = &p2, .max_new = 40, .eos = &.{}, .drafts = true });
    defer two.deinit(gpa);
    try engine.addStream(&one);
    try engine.addStream(&two);
    try engine.step();
    try std.testing.expectEqual(@as(u32, 0), one.next.?.count);
    try std.testing.expectEqual(@as(u32, 0), two.next.?.count);
    two.paused = true;
    try engine.step();
    try std.testing.expect(one.next.?.count > 0); // paused requests do not hold the bypass on
    two.paused = false;
    try engine.step();
    try std.testing.expectEqual(@as(u32, 0), one.next.?.count);
    try std.testing.expectEqual(@as(u32, 0), two.next.?.count);
    while (engine.activeCount() > 0) try engine.step();
    for ([_]*sm.Stream{ &one, &two }) |s| {
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, s.prompt());
        for (s.emitted()) |token| {
            try std.testing.expectEqual(fake.next(history.items, s.sampling, history.items.len), token);
            try history.append(gpa, token);
        }
    }
}

test "draft bypass served geometry preserves eight mixed requests through unequal endings" {
    for ([_]?Sampling{ null, .{ .seed = 17, .temperature = 0.7, .top_k = 0, .top_p = 0.95 } }) |sampling| {
        var cases: [8]Case = undefined;
        for (&cases, 0..) |*c, i| c.* = .{ .prompt = if (i % 2 == 0) &p1 else &p2, .sampling = sampling, .max_new = @intCast(3 + i * 9), .drafts = i != 3, .think_budget = if (i == 5) 9 else 0 };
        const normal = try runWithGeometry(&cases, 0, true);
        defer free(normal);
        const bypassed = try runWithGeometry(&cases, 6, true);
        defer free(bypassed);
        for (normal, bypassed) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    }
}

test "draft bypass served geometry clears backend holds and resumes at lower selected load" {
    var cfg = try servedModel();
    defer cfg.deinit(gpa);
    try cfg.setDraftBypassAt(6);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var streams: [8]sm.Stream = undefined;
    for (&streams, 0..) |*s, i| s.* = try sm.Stream.init(gpa, .{ .id = "served", .prompt = if (i % 2 == 0) &p1 else &p2, .max_new = 128, .eos = &.{}, .drafts = true });
    defer for (&streams) |*s| s.deinit(gpa);
    for (&streams) |*s| try engine.addStream(s);
    // Initial prefill heads are allowed. Inspect only the subsequent round's calls.
    for ([_]usize{ 8, 6, 2, 1, 8 }) |active| {
        for (&streams, 0..) |*s, i| s.paused = i >= active;
        const positive_before = target.positive_depth_requests;
        const zero_before = target.zero_depth_requests;
        try engine.step();
        if (active >= 6) {
            try std.testing.expectEqual(positive_before, target.positive_depth_requests);
            try std.testing.expectEqual(@as(u64, @intCast(active)), target.zero_depth_requests - zero_before);
            for (streams[0..active]) |*s| {
                try std.testing.expect(s.drafts); // request flag stays set; not proof of HIP absorption
                try std.testing.expectEqual(@as(u32, 0), s.next.?.count);
                try std.testing.expectEqual(@as(usize, 0), target.lanes.getPtr(s).?.held.items.len);
            }
        } else {
            try std.testing.expect(target.positive_depth_requests > positive_before);
            for (streams[0..active]) |*s| {
                try std.testing.expect(s.next.?.count > 0);
                try std.testing.expectEqual(@as(usize, s.next.?.count), target.lanes.getPtr(s).?.held.items.len);
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 8), target.max_verify_streams);
    try std.testing.expect(target.max_verify_rows <= 16);
    try std.testing.expect(target.max_draft_depth <= 5);
    // Cancellation releases the backend state; new admission must not inherit held data.
    engine.discard(&streams[7]);
    try std.testing.expect(!target.lanes.contains(&streams[7]));
    var replacement = try sm.Stream.init(gpa, .{ .id = "replacement", .prompt = &p2, .max_new = 5, .eos = &.{}, .drafts = true });
    defer replacement.deinit(gpa);
    try engine.addStream(&replacement);
    try engine.step();
    for (&streams) |*s| s.paused = false;
    while (engine.activeCount() > 0) try engine.step();
    for (&streams) |*s| try expectFakeHistory(s);
    try expectFakeHistory(&replacement);
}

test "draft bypass verifies one row a stream from the first shared round at the bypass load" {
    var cfg = try servedModel();
    defer cfg.deinit(gpa);
    try cfg.setDraftBypassAt(6);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var streams: [8]sm.Stream = undefined;
    for (&streams, 0..) |*s, i| s.* = try sm.Stream.init(gpa, .{ .id = "step", .prompt = if (i % 2 == 0) &p1 else &p2, .max_new = 24, .eos = &.{}, .drafts = true });
    defer for (&streams) |*s| s.deinit(gpa);
    for (&streams) |*s| try engine.addStream(s);
    // the first five joined below the load and hold drafts; the rest joined at it with none
    for (streams[0..5]) |*s| try std.testing.expect(s.next.?.count > 0);
    for (streams[5..]) |*s| try std.testing.expectEqual(@as(u32, 0), s.next.?.count);
    try engine.step();
    try std.testing.expectEqual(@as(usize, 8), target.max_verify_rows);
    for (&streams) |*s| try std.testing.expectEqual(@as(usize, 2), s.emitted().len);
    while (engine.activeCount() > 0) try engine.step();
    for (&streams) |*s| try expectFakeHistory(s);
}

fn expectFakeHistory(s: *sm.Stream) !void {
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, s.prompt());
    for (s.emitted()) |token| {
        try std.testing.expectEqual(fake.next(history.items, s.sampling, history.items.len), token);
        try history.append(gpa, token);
    }
}

test "draft bypass refuses unsupported backends and invalid thresholds" {
    var cfg = try model();
    defer cfg.deinit(gpa);
    try cfg.setDraftBypassAt(0);
    try std.testing.expectError(error.BadDraftBypass, cfg.setDraftBypassAt(2)); // no node probabilities
    cfg.node_probabilities = true;
    try std.testing.expectError(error.BadDraftBypass, cfg.setDraftBypassAt(1));
    try std.testing.expectError(error.BadDraftBypass, cfg.setDraftBypassAt(cfg.batch_streams + 1));
    try cfg.setDraftBypassAt(2);
    try std.testing.expectEqual(@as(u32, 2), cfg.draft_bypass_at);
    try cfg.setDraftBypassAt(0);
}

test "drafted rounds commit the one-token decode, greedy and sampled" {
    for ([_]?Sampling{ null, .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 } }) |s| {
        const drafted = try run(&.{.{ .prompt = &p1, .sampling = s }});
        defer free(drafted);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }});
        defer free(plain);
        try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
        // and the fake target's own decode
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, &p1);
        for (drafted[0]) |t| {
            try std.testing.expectEqual(fake.next(history.items, s, history.items.len), t);
            try history.append(gpa, t);
        }
    }
}

test "shared rounds commit what each stream commits alone" {
    const sampled: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
    const together = try run(&.{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30 } });
    defer free(together);
    const one = try run(&.{.{ .prompt = &p1 }});
    defer free(one);
    const two = try run(&.{.{ .prompt = &p2, .sampling = sampled, .max_new = 30 }});
    defer free(two);
    try std.testing.expectEqualSlices(u32, one[0], together[0]);
    try std.testing.expectEqualSlices(u32, two[0], together[1]);
}

test "the thinking budget's forced close is the same drafted, plain and shared" {
    const drafted = try run(&.{.{ .prompt = &p2, .think_budget = 9 }});
    defer free(drafted);
    const plain = try run(&.{.{ .prompt = &p2, .think_budget = 9, .drafts = false }});
    defer free(plain);
    const shared = try run(&.{ .{ .prompt = &p2, .think_budget = 9 }, .{ .prompt = &p1, .drafts = false } });
    defer free(shared);
    try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
    try std.testing.expectEqualSlices(u32, plain[0], shared[0]);
    try std.testing.expectEqual(@as(u32, 90), drafted[0][8]);
}

test "the loop guard matches drafted and plain, reports its period, and answers after close" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2 }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), drafted[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .stop);
    try std.testing.expectEqualSlices(u32, &.{ 90, 91, 92, 40, 41, 42, 96 }, plain[0].emitted[329..]);

    const off = try runGuard(&.{.{ .prompt = &p2, .loop_guard = false, .think_open = true }}, 5000);
    defer freeGuard(off);
    try std.testing.expect(off[0].finished and off[0].reason == .length and off[0].emitted.len == 400);
}

test "a cycle in the answer does not refire or reclose" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), plain[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .length);
    var close_count: usize = 0;
    for (plain[0].emitted) |token| close_count += @intFromBool(token == 90);
    try std.testing.expectEqual(@as(usize, 1), close_count);
}

/// The rule's learned overhead entries after one drafted request, with or without frozen timing.
fn overheadsAfter(frozen: bool) !usize {
    var cfg = try model();
    defer cfg.deinit(gpa);
    cfg.frozen_timing = frozen;
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer p.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 1, 2, 3, 4, 5, 6 }, .max_new = 40, .eos = &.{96}, .sampling = null, .drafts = true, .proposer = p.proposer(), .think_budget = 0, .think_close = &.{ 90, 91, 92 }, .think_end = 91 });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    return engine.rule.overhead.items.len;
}

test "frozen timing: the round loop learns no overhead from the clock" {
    try std.testing.expect(try overheadsAfter(false) > 0);
    try std.testing.expectEqual(@as(usize, 0), try overheadsAfter(true));
}
