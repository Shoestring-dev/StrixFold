//! Caps generation requests waiting for a lane and the bytes in flight; past a cap a request gets 429 and Retry-After.
const std = @import("std");
const errors = @import("errors.zig");
const Cx = errors.Cx;
const log = @import("log.zig");
const ArenaAllocator = std.heap.ArenaAllocator;

pub const Limits = struct {
    /// Requests that may be in flight beyond the engine's lanes (TF_QUEUE_MAX); null: no count cap.
    waiting: ?u32 = 64,
    /// Bytes the requests in flight may hold together when one more would wait (TF_QUEUE_MIB); null: no byte cap.
    bytes: ?u64 = 4096 << 20,
    /// The Retry-After a refusal sends, in seconds (TF_QUEUE_RETRY_S).
    retry_s: u32 = 10,

    /// TF_QUEUE_MAX, TF_QUEUE_MIB, TF_QUEUE_RETRY_S from the environment.
    pub fn fromEnv() Limits {
        return parse(env("TF_QUEUE_MAX"), env("TF_QUEUE_MIB"), env("TF_QUEUE_RETRY_S"));
    }

    /// Each a whole number or "off" (no cap; Retry-After has none); unset, empty or unreadable keeps the default.
    pub fn parse(max: ?[]const u8, mib: ?[]const u8, retry: ?[]const u8) Limits {
        var l: Limits = .{};
        if (max) |v| if (isOff(v)) {
            l.waiting = null;
        } else if (std.fmt.parseInt(u32, v, 10)) |n| {
            l.waiting = n;
        } else |_| {};
        if (mib) |v| if (isOff(v)) {
            l.bytes = null;
        } else if (std.fmt.parseInt(u32, v, 10)) |n| {
            l.bytes = @as(u64, n) << 20;
        } else |_| {};
        if (retry) |v| if (std.fmt.parseInt(u32, v, 10)) |n| {
            l.retry_s = @max(1, n);
        } else |_| {};
        return l;
    }

    fn isOff(v: []const u8) bool {
        return std.mem.eql(u8, v, "off");
    }

    fn env(name: [*:0]const u8) ?[]const u8 {
        const v = std.c.getenv(name) orelse return null;
        const s = std.mem.span(v);
        return if (s.len == 0) null else s;
    }
};

/// The server's count of generation requests in flight and the bytes they hold.
pub const Gate = struct {
    limits: Limits = .{},
    /// The engine's lanes: this many requests in flight never wait, and are never refused.
    lanes: u32 = 1,
    count: std.atomic.Value(u32) = .init(0),
    bytes: std.atomic.Value(u64) = .init(0),
    /// Requests refused so far, at the head (the count) and once rendered (the bytes).
    refused_count: std.atomic.Value(u64) = .init(0),
    refused_bytes: std.atomic.Value(u64) = .init(0),

    /// Admits one unless lanes + waiting are in flight; holds ``t`` to ``t.leave()``; charges ``arena`` (null: none).
    pub fn enter(g: *Gate, t: *Ticket, arena: ?*const ArenaAllocator) bool {
        var n = g.count.load(.acquire);
        while (true) {
            if (g.limits.waiting) |w| if (n >= g.lanes +| w) {
                g.note(&g.refused_count, "in flight");
                return false;
            };
            n = g.count.cmpxchgWeak(n, n + 1, .acq_rel, .acquire) orelse break;
        }
        t.* = .{ .gate = g, .arena = arena, .prev = current };
        current = t;
        return true;
    }

    /// Counts a refusal, logged at the 1st, 2nd, 4th, 8th... of its kind so a flood writes few lines.
    fn note(g: *Gate, counter: *std.atomic.Value(u64), what: []const u8) void {
        const n = counter.fetchAdd(1, .monotonic) + 1;
        if (!std.math.isPowerOfTwo(n)) return;
        log.line("queue cap: {d} requests refused at the {s} cap so far; {d} in flight ({d} lanes), {d} MiB held", .{ n, what, g.count.load(.acquire), g.lanes, g.bytes.load(.acquire) >> 20 });
    }
};

/// One admitted request; on its own thread, ``current`` names it so the render can charge it.
pub const Ticket = struct {
    gate: *Gate,
    arena: ?*const ArenaAllocator,
    bytes: u64 = 0,
    prev: ?*Ticket = null,

    /// Counts the arena's capacity (never less); false: the request would wait and the in-flight bytes pass the cap.
    pub fn charge(t: *Ticket) bool {
        const g = t.gate;
        const now: u64 = if (t.arena) |ar| ar.queryCapacity() else 0;
        if (now > t.bytes) {
            _ = g.bytes.fetchAdd(now - t.bytes, .acq_rel);
            t.bytes = now;
        }
        const cap = g.limits.bytes orelse return true;
        if (g.count.load(.acquire) <= g.lanes) return true;
        if (g.bytes.load(.acquire) <= cap) return true;
        g.note(&g.refused_bytes, "held");
        return false;
    }

    /// Gives back the request's place and bytes; every admitted request calls it once, however its reply ends.
    pub fn leave(t: *Ticket) void {
        _ = t.gate.bytes.fetchSub(t.bytes, .acq_rel);
        t.bytes = 0;
        _ = t.gate.count.fetchSub(1, .acq_rel);
        if (current == t) current = t.prev;
    }
};

/// The request this thread serves now, admitted by ``Gate.enter``; null outside a generation route.
threadlocal var current: ?*Ticket = null;

/// chat.prepare's check once rendered: charges the request, 429 past the byte cap if it would wait (no ticket: none).
pub fn charge(cx: *Cx) errors.Refused!void {
    const t = current orelse return;
    if (t.charge()) return;
    const g = t.gate;
    return cx.fail(.queue_full, "the server's requests in flight hold {d} MiB, past its cap of {d} MiB; retry after {d} s", .{ g.bytes.load(.acquire) >> 20, (g.limits.bytes orelse 0) >> 20, g.limits.retry_s });
}

/// The message of a request refused at its head: lanes + waiting are in flight already.
pub fn fullMessage(a: std.mem.Allocator, g: *const Gate) std.mem.Allocator.Error![]const u8 {
    return std.fmt.allocPrint(a, "the server has {d} requests in flight, {d} lanes and {d} waiting, its cap; retry after {d} s", .{ g.count.load(.acquire), g.lanes, g.limits.waiting orelse 0, g.limits.retry_s });
}
