//! Host memory reporting, isolated request refusal, the pool-full retry and sliced prompt passes through LaneHost.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Memory = api.Memory;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

test "a lane host reports its backend's memory counts, and none without them" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try std.testing.expect(host.engine().memory(false) == null);
    const Counts = struct {
        resets: u32 = 0,
        fn read(ctx: ?*anyopaque, reset_peak: bool) ?Memory {
            const c: *@This() = @ptrCast(@alignCast(ctx.?));
            if (reset_peak) c.resets += 1;
            return .{ .active = 5, .peak = if (reset_peak) 5 else 9 };
        }
    };
    var counts: Counts = .{};
    host.memory = .{ .ctx = &counts, .read = Counts.read };
    try std.testing.expectEqual(@as(u64, 9), host.engine().memory(false).?.peak);
    try std.testing.expectEqual(@as(u64, 5), host.engine().memory(true).?.peak);
    try std.testing.expectEqual(@as(u32, 1), counts.resets);
}

test "a request the backend refuses fails alone, in the backend's words" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .refuse_sampled = true };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Words = struct {
        fn text(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return if (err == error.SamplingRefused) "send temperature 0" else null;
        }
    };
    host.explain = .{ .text = Words.text };
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        message: []const u8 = "",
        tokens: usize = 0,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| {
                    b.done = f.reason;
                    b.message = f.message;
                },
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var plain: Box = .{};
    var sampled: Box = .{};
    const greedy: Request = .{ .prompt = &prompt, .max_tokens = 64 };
    const keyed: Request = .{ .prompt = &prompt, .max_tokens = 64, .sampling = .{ .seed = 3, .temperature = 0.7, .top_k = 5 } };
    const e = host.engine();
    try e.submit(1, &greedy, .{ .ctx = &plain, .event = Box.event });
    try e.submit(2, &keyed, .{ .ctx = &sampled, .event = Box.event });
    try std.testing.expectEqual(Reason.failed, sampled.wait());
    try std.testing.expectEqualStrings("send temperature 0", sampled.message);
    try std.testing.expectEqual(Reason.length, plain.wait());
    try std.testing.expectEqual(@as(usize, 64), plain.tokens);
}

test "a request refused for room while another stream runs is tried again within a second" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .refuse_full = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        tokens: usize = 0,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn state(b: *@This()) struct { ?Reason, usize } {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return .{ b.done, b.tokens };
        }
        // the reason the request ended with, or null when it has not ended within `ms`
        fn waitFor(b: *@This(), ms: u32) ?Reason {
            for (0..ms) |_| {
                if (b.state()[0]) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
            return b.state()[0];
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var long: Box = .{};
    var short: Box = .{};
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 1 << 30 }, .{ .ctx = &long, .event = Box.event });
    while (long.state()[1] == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    // the second request's prompt pass is refused once for room (error.ContextFull) while the first stream runs on:
    // it goes back to the queue and is tried again within a second, not only when that stream ends
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 8 }, .{ .ctx = &short, .event = Box.event });
    try std.testing.expectEqual(@as(?Reason, Reason.length), short.waitFor(5000));
    try std.testing.expectEqual(@as(usize, 8), short.state()[1]);
    try std.testing.expectEqual(@as(usize, 0), target.refuse_full);
    try std.testing.expect(long.state()[0] == null);
    e.cancel(1);
    try std.testing.expect(long.waitFor(5000) != null);
}

test "after a long prompt pass, a stream near its max_tokens ends before the next request's pass" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const Slow = struct {
        // each prompt pass takes 1.1 s, a long pass for the host (its whole pass runs inside the admission)
        fn hook(_: *anyopaque, _: *lanes.stream.Stream, _: usize) void {
            std.Io.sleep(std.testing.io, .fromMilliseconds(1100), .awake) catch {};
        }
    };
    var unused: u8 = 0;
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_hook = Slow.hook, .prefill_hook_ctx = &unused };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Seq = struct {
        mutex: std.Io.Mutex = .init,
        n: u32 = 0,
        a_done: ?u32 = null,
        b_first: ?u32 = null,
        b_done: bool = false,
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const q: *@This() = @ptrCast(@alignCast(ctx));
            q.mutex.lockUncancelable(std.testing.io);
            defer q.mutex.unlock(std.testing.io);
            q.n += 1;
            switch (e.*) {
                .tokens => if (id == 2 and q.b_first == null) {
                    q.b_first = q.n;
                },
                .finished => if (id == 1) {
                    q.a_done = q.n;
                } else {
                    q.b_done = true;
                },
                else => {},
            }
        }
        fn done(q: *@This()) bool {
            q.mutex.lockUncancelable(std.testing.io);
            defer q.mutex.unlock(std.testing.io);
            return q.a_done != null and q.b_done;
        }
    };
    var seq: Seq = .{};
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 64 }, .{ .ctx = &seq, .event = Seq.event });
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 64 }, .{ .ctx = &seq, .event = Seq.event });
    for (0..10000) |_| {
        if (seq.done()) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(seq.done());
    // the first request (64 tokens: within finish_left) ended before the second's first token
    try std.testing.expect(seq.a_done.? < seq.b_first.?);
}

/// Each request's tokens and end, and each prompt pass's round count at every one of its chunks (in order of passes).
const Slices = struct {
    mutex: std.Io.Mutex = .init,
    target: *lanes.fake.Fake,
    streams: [2]?*lanes.stream.Stream = .{ null, null },
    rounds: [2][8]u64 = undefined,
    chunks: [2]usize = .{ 0, 0 },
    tokens: [3]std.ArrayList(u32) = .{ .empty, .empty, .empty },
    done: [3]?Reason = .{ null, null, null },

    fn hook(ctx: *anyopaque, s: *lanes.stream.Stream, _: usize) void {
        const q: *Slices = @ptrCast(@alignCast(ctx));
        q.mutex.lockUncancelable(std.testing.io);
        defer q.mutex.unlock(std.testing.io);
        const k: usize = if (q.streams[0] == null or q.streams[0] == s) 0 else 1;
        q.streams[k] = s;
        if (q.chunks[k] < 8) q.rounds[k][q.chunks[k]] = q.target.rounds;
        q.chunks[k] += 1;
    }
    fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
        const q: *Slices = @ptrCast(@alignCast(ctx));
        q.mutex.lockUncancelable(std.testing.io);
        defer q.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| q.tokens[@intCast(id)].appendSlice(std.testing.allocator, t) catch {},
            .finished => |f| q.done[@intCast(id)] = f.reason,
            else => {},
        }
    }
    fn count(q: *Slices, id: usize) usize {
        q.mutex.lockUncancelable(std.testing.io);
        defer q.mutex.unlock(std.testing.io);
        return q.tokens[id].items.len;
    }
    fn waitDone(q: *Slices, id: usize) ?Reason {
        for (0..5000) |_| {
            q.mutex.lockUncancelable(std.testing.io);
            const d = q.done[id];
            q.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return null;
    }
    fn deinit(q: *Slices) void {
        for (&q.tokens) |*t| t.deinit(std.testing.allocator);
    }
};

/// The fake's greedy reply to `prompt`: its draws from the history each token extends.
fn fakeReply(prompt: []const u32, comptime n: usize) [n]u32 {
    var hist: [64]u32 = undefined;
    @memcpy(hist[0..prompt.len], prompt);
    var out: [n]u32 = undefined;
    for (&out, 0..) |*t, i| {
        t.* = lanes.fake.next(hist[0 .. prompt.len + i], null, prompt.len + i);
        hist[prompt.len + i] = t.*;
    }
    return out;
}

fn slicedPass(slice_ns: u64) !struct { [8]u64, [8]u64, usize } {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 6, .slice_chunks = 1 };
    defer target.deinit();
    var q: Slices = .{ .target = &target };
    defer q.deinit();
    target.prefill_hook = Slices.hook;
    target.prefill_hook_ctx = &q;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .fill_slice_ns = slice_ns });
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 1 << 30 }, .{ .ctx = &q, .event = Slices.event });
    while (q.count(1) < 3) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 4 }, .{ .ctx = &q, .event = Slices.event });
    try std.testing.expectEqual(@as(?Reason, Reason.length), q.waitDone(2));
    try std.testing.expectEqualSlices(u32, &fakeReply(&prompt, 4), q.tokens[2].items);
    e.cancel(1);
    try std.testing.expectEqual(@as(?Reason, Reason.cancelled), q.waitDone(1));
    try std.testing.expectEqual(@as(usize, 6), q.chunks[0]);
    try std.testing.expectEqual(@as(usize, 6), q.chunks[1]);
    return .{ q.rounds[0], q.rounds[1], target.lanes.count() };
}

test "a prompt pass reads a slice, then the decoding stream's round runs; alone it reads whole" {
    const alone, const beside, const left = try slicedPass(std.time.ns_per_s);
    // the first request's pass ran alone: its six chunks before any round
    try std.testing.expectEqualSlices(u64, &.{ 0, 0, 0, 0, 0, 0 }, alone[0..6]);
    // the second's ran beside a decoding stream: one chunk a slice, a round between each two
    for (1..6) |i| try std.testing.expect(beside[i] > beside[i - 1]);
    try std.testing.expectEqual(@as(usize, 0), left);
}

test "without a slice time a prompt pass reads whole beside a decoding stream" {
    _, const beside, _ = try slicedPass(0);
    for (1..6) |i| try std.testing.expectEqual(beside[0], beside[i]);
}

test "a paused prompt pass cancelled between slices frees its lane; the decoding stream goes on" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 200, .slice_chunks = 1 };
    defer target.deinit();
    var q: Slices = .{ .target = &target };
    defer q.deinit();
    target.prefill_hook = Slices.hook;
    target.prefill_hook_ctx = &q;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .fill_slice_ns = std.time.ns_per_s });
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 1 << 30 }, .{ .ctx = &q, .event = Slices.event });
    while (q.count(1) < 3) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 4 }, .{ .ctx = &q, .event = Slices.event });
    for (0..5000) |_| {
        q.mutex.lockUncancelable(std.testing.io);
        const n = q.chunks[1];
        q.mutex.unlock(std.testing.io);
        if (n >= 3) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    e.cancel(2);
    try std.testing.expectEqual(@as(?Reason, Reason.cancelled), q.waitDone(2));
    try std.testing.expectEqual(@as(usize, 0), q.count(2));
    const before = q.count(1);
    while (q.count(1) < before + 3) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    e.cancel(1);
    try std.testing.expectEqual(@as(?Reason, Reason.cancelled), q.waitDone(1));
    try std.testing.expect(q.chunks[1] < 200);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

test "stopping the host while a prompt pass is paused frees every lane" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 200, .slice_chunks = 1 };
    defer target.deinit();
    var q: Slices = .{ .target = &target };
    defer q.deinit();
    target.prefill_hook = Slices.hook;
    target.prefill_hook_ctx = &q;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .fill_slice_ns = std.time.ns_per_s });
    try host.start();
    errdefer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 1 << 30 }, .{ .ctx = &q, .event = Slices.event });
    for (0..5000) |_| {
        if (q.count(1) >= 3) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 4 }, .{ .ctx = &q, .event = Slices.event });
    for (0..5000) |_| {
        q.mutex.lockUncancelable(std.testing.io);
        const n = q.chunks[1];
        q.mutex.unlock(std.testing.io);
        if (n >= 3) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    host.stop(); // the second request's pass paused between slices, the first decoding
    try std.testing.expect(q.chunks[1] < 200);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}
