//! The queue cap (queue_cap.zig): 429 past either cap, and the bytes given back however a reply ends.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const queue_cap = @import("queue_cap.zig");
const Conn = @import("http_conn.zig").Conn;

/// Bytes as tokens; every prompt renders as "prompt".
const FixedText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// One lane; answers "ok" at once, or fails every submit (.busy). At submit it notes the gate's count and bytes.
const OkEngine = struct {
    busy: bool = false,
    gate: ?*queue_cap.Gate = null,
    submits: usize = 0,
    count_at_submit: u32 = 0,
    bytes_at_submit: u64 = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *OkEngine = @ptrCast(@alignCast(ctx));
        e.submits += 1;
        if (e.gate) |g| {
            e.count_at_submit = g.count.load(.acquire);
            e.bytes_at_submit = g.bytes.load(.acquire);
        }
        if (e.busy) return error.Busy;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'o', 'k' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const chat_body = "{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";

/// The reply to a request that declares ``length`` body bytes and sends only ``part`` of them.
fn postHead(a: std.mem.Allocator, srv: *server.Server, path: []const u8, part: []const u8, length: usize) ![]u8 {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n{s}", .{ path, length, part });
    if (std.c.write(pair[0], head.ptr, head.len) != head.len) return error.Closed;
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    conn.arena = &arena;
    conn.retry_after_s = srv.queue.limits.retry_s;
    if (try conn.readRequest(arena.allocator(), false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, arena.allocator());
    var tmp: [8192]u8 = undefined;
    const n = std.posix.read(pair[0], &tmp) catch 0;
    return a.dupe(u8, tmp[0..n]);
}

/// One request's whole HTTP reply at ``path`` via a server arena; ``drop``: the client shuts its side after sending.
fn post(a: std.mem.Allocator, srv: *server.Server, path: []const u8, body: []const u8, drop: bool) ![]u8 {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{ path, body.len });
    for ([_][]const u8{ head, body }) |part| {
        var sent: usize = 0;
        while (sent < part.len) {
            const n = std.c.write(pair[0], part[sent..].ptr, part.len - sent);
            if (n <= 0) return error.Closed;
            sent += @intCast(n);
        }
    }
    if (drop) _ = std.c.shutdown(pair[0], std.posix.SHUT.WR);
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    conn.arena = &arena;
    conn.retry_after_s = srv.queue.limits.retry_s;
    if (try conn.readRequest(arena.allocator(), false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, arena.allocator());
    var got: std.ArrayList(u8) = .empty;
    var tmp: [4096]u8 = undefined;
    var pollfd = [_]std.posix.pollfd{.{ .fd = pair[0], .events = std.posix.POLL.IN, .revents = 0 }};
    while (try std.posix.poll(&pollfd, 200) > 0) {
        const n = std.posix.read(pair[0], &tmp) catch break;
        if (n == 0) break;
        try got.appendSlice(a, tmp[0..n]);
    }
    return got.items;
}

const Fixture = struct {
    text: FixedText = .{},
    backend: OkEngine = .{},
    srv: *server.Server = undefined,

    fn start(f: *Fixture, limits: queue_cap.Limits) !void {
        f.srv = try server.Server.init(std.testing.allocator, std.testing.io, f.backend.engine(), f.text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false, .enable_thinking = false, .queue = limits }, null);
        f.backend.gate = &f.srv.queue;
    }

    fn stop(f: *Fixture) void {
        f.srv.deinit();
    }

    fn expectEmpty(f: *Fixture) !void {
        try std.testing.expectEqual(@as(u32, 0), f.srv.queue.count.load(.acquire));
        try std.testing.expectEqual(@as(u64, 0), f.srv.queue.bytes.load(.acquire));
    }

    /// Fills every place the cap has (lanes + waiting) by hand, proving each was free, and then gives them back.
    fn expectAllPlacesFree(f: *Fixture) !void {
        const g = &f.srv.queue;
        var tickets: [16]queue_cap.Ticket = undefined;
        const places = g.lanes + g.limits.waiting.?;
        for (tickets[0..places]) |*t| try std.testing.expect(g.enter(t, null));
        var extra: queue_cap.Ticket = undefined;
        try std.testing.expect(!g.enter(&extra, null));
        var i = places;
        while (i > 0) : (i -= 1) tickets[i - 1].leave();
        try f.expectEmpty();
    }
};

fn code(reply: []const u8) []const u8 {
    return if (reply.len >= 12) reply[9..12] else "";
}

test "Limits: the defaults, whole numbers, off, and unreadable values" {
    const d = queue_cap.Limits.parse(null, null, null);
    try std.testing.expectEqual(@as(?u32, 64), d.waiting);
    try std.testing.expectEqual(@as(?u64, 4096 << 20), d.bytes);
    try std.testing.expectEqual(@as(u32, 10), d.retry_s);
    const set = queue_cap.Limits.parse("8", "512", "3");
    try std.testing.expectEqual(@as(?u32, 8), set.waiting);
    try std.testing.expectEqual(@as(?u64, 512 << 20), set.bytes);
    try std.testing.expectEqual(@as(u32, 3), set.retry_s);
    const off = queue_cap.Limits.parse("off", "off", "0");
    try std.testing.expectEqual(@as(?u32, null), off.waiting);
    try std.testing.expectEqual(@as(?u64, null), off.bytes);
    try std.testing.expectEqual(@as(u32, 1), off.retry_s);
    const bad = queue_cap.Limits.parse("-1", "lots", "x");
    try std.testing.expectEqual(d.waiting, bad.waiting);
    try std.testing.expectEqual(d.bytes, bad.bytes);
    try std.testing.expectEqual(d.retry_s, bad.retry_s);
}

test "the gate admits lanes + waiting requests and refuses the next until one leaves" {
    var g: queue_cap.Gate = .{ .lanes = 2, .limits = .{ .waiting = 3 } };
    var t: [6]queue_cap.Ticket = undefined;
    for (t[0..5]) |*x| try std.testing.expect(g.enter(x, null));
    try std.testing.expect(!g.enter(&t[5], null));
    try std.testing.expectEqual(@as(u64, 1), g.refused_count.load(.acquire));
    t[4].leave();
    try std.testing.expect(g.enter(&t[5], null));
    t[5].leave();
    var i: usize = 4;
    while (i > 0) : (i -= 1) t[i - 1].leave();
    try std.testing.expectEqual(@as(u32, 0), g.count.load(.acquire));
    var off: queue_cap.Gate = .{ .lanes = 1, .limits = .{ .waiting = null } };
    var many: [40]queue_cap.Ticket = undefined;
    for (&many) |*x| try std.testing.expect(off.enter(x, null));
    i = many.len;
    while (i > 0) : (i -= 1) many[i - 1].leave();
}

test "charge counts each request arena's capacity, never less than before, and leave gives it back" {
    var g: queue_cap.Gate = .{ .lanes = 4 };
    var a1 = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer a1.deinit();
    var a2 = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer a2.deinit();
    _ = try a1.allocator().alloc(u8, 1 << 20);
    _ = try a2.allocator().alloc(u8, 3 << 20);
    var t1: queue_cap.Ticket = undefined;
    var t2: queue_cap.Ticket = undefined;
    try std.testing.expect(g.enter(&t1, &a1));
    try std.testing.expect(g.enter(&t2, &a2));
    try std.testing.expect(t1.charge());
    try std.testing.expect(t2.charge());
    try std.testing.expectEqual(@as(u64, a1.queryCapacity() + a2.queryCapacity()), g.bytes.load(.acquire));
    _ = try a1.allocator().alloc(u8, 2 << 20);
    try std.testing.expect(t1.charge());
    try std.testing.expect(t1.charge()); // charged twice: counted once
    try std.testing.expectEqual(@as(u64, a1.queryCapacity() + a2.queryCapacity()), g.bytes.load(.acquire));
    t2.leave();
    try std.testing.expectEqual(@as(u64, a1.queryCapacity()), g.bytes.load(.acquire));
    t1.leave();
    try std.testing.expectEqual(@as(u64, 0), g.bytes.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), g.count.load(.acquire));
}

test "the byte cap refuses only a request that would wait: never one a lane can take" {
    var g: queue_cap.Gate = .{ .lanes = 2, .limits = .{ .bytes = 1 } };
    var a1 = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer a1.deinit();
    _ = try a1.allocator().alloc(u8, 1 << 20);
    var t: [3]queue_cap.Ticket = undefined;
    try std.testing.expect(g.enter(&t[0], &a1));
    try std.testing.expect(t[0].charge()); // over the cap, but within the lanes
    try std.testing.expect(g.enter(&t[1], &a1));
    try std.testing.expect(t[1].charge());
    try std.testing.expect(g.enter(&t[2], null)); // a third would wait: refused once charged
    try std.testing.expect(!t[2].charge());
    try std.testing.expectEqual(@as(u64, 1), g.refused_bytes.load(.acquire));
    t[2].leave();
    t[1].leave();
    t[0].leave();
}

test "at the defaults a server whose requests never outnumber its lanes refuses none, however much they hold" {
    // as behind a router that sends a server at most its /health max_batch_size requests at once
    var g: queue_cap.Gate = .{ .lanes = 8 };
    try std.testing.expectEqual(queue_cap.Limits.parse(null, null, null), g.limits);
    g.bytes.store(5 << 30, .release); // as if long prompts in flight held 5 GiB, past the 4 GiB cap
    var t: [8]queue_cap.Ticket = undefined;
    for (&t) |*x| {
        try std.testing.expect(g.enter(x, null));
        try std.testing.expect(x.charge());
    }
    var extra: queue_cap.Ticket = undefined; // the ninth would wait: the byte cap now refuses it
    try std.testing.expect(g.enter(&extra, null));
    try std.testing.expect(!extra.charge());
    extra.leave();
    var i: usize = t.len;
    while (i > 0) : (i -= 1) t[i - 1].leave();
    try std.testing.expectEqual(@as(u32, 0), g.count.load(.acquire));
}

test "a chat through the route holds its place and bytes while submitted and gives them back when it finishes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f: Fixture = .{};
    try f.start(.{ .waiting = 2 });
    defer f.stop();
    const reply = try post(arena.allocator(), f.srv, "/v1/chat/completions", chat_body, false);
    try std.testing.expectEqualStrings("200", code(reply));
    try std.testing.expectEqual(@as(u32, 1), f.backend.count_at_submit);
    try std.testing.expect(f.backend.bytes_at_submit > chat_body.len);
    try f.expectEmpty();
    try f.expectAllPlacesFree();
}

test "a request whose client has gone gives its place back" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f: Fixture = .{};
    try f.start(.{ .waiting = 2 });
    defer f.stop();
    _ = try post(arena.allocator(), f.srv, "/v1/chat/completions", chat_body, true);
    try std.testing.expectEqual(@as(usize, 0), f.backend.submits);
    try f.expectEmpty();
    try f.expectAllPlacesFree();
}

test "a request that fails (a refused body, an engine that refuses the submit) gives its place back" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f: Fixture = .{ .backend = .{ .busy = true } };
    try f.start(.{ .waiting = 2 });
    defer f.stop();
    const bad = try post(arena.allocator(), f.srv, "/v1/chat/completions", "{not json", false);
    try std.testing.expectEqualStrings("400", code(bad));
    try f.expectEmpty();
    const busy = try post(arena.allocator(), f.srv, "/v1/chat/completions", chat_body, false);
    try std.testing.expect(!std.mem.eql(u8, code(busy), "200")); // the engine's Busy, refused
    try std.testing.expectEqual(@as(usize, 1), f.backend.submits);
    try f.expectEmpty();
    try f.expectAllPlacesFree();
}

test "past lanes + waiting a request is refused with 429 and Retry-After before its body is parsed, its connection kept" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = .{};
    try f.start(.{ .waiting = 2, .retry_s = 7 });
    defer f.stop();
    var held: [3]queue_cap.Ticket = undefined; // one lane and two waiting
    for (&held) |*t| try std.testing.expect(f.srv.queue.enter(t, null));
    const reply = try post(a, f.srv, "/v1/chat/completions", chat_body, false);
    try std.testing.expectEqualStrings("429", code(reply));
    try std.testing.expect(std.mem.indexOf(u8, reply, "\r\nRetry-After: 7\r\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, reply, "\r\nConnection: close\r\n") == null); // a short body is drained
    try std.testing.expect(std.mem.indexOf(u8, reply, "\"code\": \"rate_limit_exceeded\"") != null);
    const messages = try post(a, f.srv, "/v1/messages", chat_body, false);
    try std.testing.expectEqualStrings("429", code(messages));
    try std.testing.expect(std.mem.indexOf(u8, messages, "rate_limit_error") != null);
    try std.testing.expectEqual(@as(usize, 0), f.backend.submits);
    try std.testing.expectEqual(@as(u64, 2), f.srv.queue.refused_count.load(.acquire));
    // a body past drain_max (here declared, never sent) is left unread: the 429 closes the connection
    const long_head = "{\"model\":\"m\"}";
    const long = try postHead(a, f.srv, "/v1/chat/completions", long_head, 2 << 20);
    try std.testing.expectEqualStrings("429", code(long));
    try std.testing.expect(std.mem.indexOf(u8, long, "\r\nConnection: close\r\n") != null);
    held[2].leave();
    const admitted = try post(a, f.srv, "/v1/chat/completions", chat_body, false);
    try std.testing.expectEqualStrings("200", code(admitted));
    held[1].leave();
    held[0].leave();
    try f.expectEmpty();
}

test "a request that would wait while the requests in flight hold past the byte cap is refused with 429 before its reply opens" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = .{};
    try f.start(.{ .waiting = 4, .bytes = 1 << 20 });
    defer f.stop();
    var running = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer running.deinit();
    _ = try running.allocator().alloc(u8, 2 << 20);
    var lane: queue_cap.Ticket = undefined; // the one lane, running a request that holds 2 MiB
    try std.testing.expect(f.srv.queue.enter(&lane, &running));
    try std.testing.expect(lane.charge());
    const held = f.srv.queue.bytes.load(.acquire);
    const streamed = "{\"model\":\"m\",\"max_tokens\":2,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    const reply = try post(a, f.srv, "/v1/chat/completions", streamed, false);
    try std.testing.expectEqualStrings("429", code(reply));
    try std.testing.expect(std.mem.indexOf(u8, reply, "\r\nRetry-After: 10\r\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, reply, "text/event-stream") == null);
    try std.testing.expectEqual(@as(usize, 0), f.backend.submits);
    try std.testing.expectEqual(@as(u64, 1), f.srv.queue.refused_bytes.load(.acquire));
    try std.testing.expectEqual(held, f.srv.queue.bytes.load(.acquire));
    lane.leave();
    try f.expectEmpty();
}
