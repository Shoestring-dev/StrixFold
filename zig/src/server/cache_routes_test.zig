//! The /cache routes on a fake engine: absent without the key, the bearer, the cap, the MAC, the parser, a round trip.
const std = @import("std");
const api = @import("engine_api");
const wire = api.cache_wire;
const server = @import("root.zig");
const cache_key = @import("cache_key.zig");
const routes = @import("routes.zig");
const DropLate = @import("late_system_test.zig").DropLate;
const Conn = @import("http_conn.zig").Conn;
const gpa = std.testing.allocator;

const shape: wire.Shape = .{ .ns = @splat(5), .block_tokens = 2, .max_blocks = 4, .max_positions = 8, .slot_bytes = 3, .block_bytes = 4 };
const key_text = "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c\nsk-move\n";
const probe_body =
    \\{"model":"m","messages":[{"role":"user","content":"hello"}]}
;

/// An engine that keeps one state in host memory: `kept` ids at `at`; its parts count up from 1.
const Mover = struct {
    prompt: []u32 = &.{},
    kept: ?[]u32 = null,
    at: u32 = 0,
    imported: usize = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .cache = cache, .cache_shape = cacheShape } };
    }
    fn of(ctx: *anyopaque) *Mover {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn submit(_: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn cacheShape(_: *anyopaque) ?api.CacheShape {
        return shape;
    }
    fn covers(e: *Mover, prompt: []const u32) bool {
        const k = e.kept orelse return false;
        return prompt.len >= k.len and std.mem.eql(u32, prompt[0..k.len], k);
    }
    fn cache(ctx: *anyopaque, op: *api.CacheOp) void {
        const e = of(ctx);
        defer op.done = true;
        if (op.kind != .import_state) {
            gpa.free(e.prompt);
            e.prompt = gpa.dupe(u32, op.prompt) catch &.{};
        }
        switch (op.kind) {
            .probe => op.tokens = if (e.covers(op.prompt)) e.at else 0,
            .export_state => {
                if (!e.covers(op.prompt)) {
                    op.err = error.NoState;
                    return;
                }
                const ids = op.gpa.dupe(u32, e.kept.?) catch return;
                const slot = op.gpa.alloc(u8, shape.slot_bytes) catch return;
                for (slot, 1..) |*b, i| b.* = @intCast(i);
                const rows = op.gpa.alloc(u8, shape.block_bytes * (e.at / shape.block_tokens)) catch return;
                for (rows, 1..) |*b, i| b.* = @intCast(i);
                op.out = .{ .at = e.at, .ids = ids, .parts = .{ .slot = slot, .rows = rows } };
                op.tokens = e.at;
            },
            .import_state => {
                if (op.at % shape.block_tokens != 0 or op.slot.len != shape.slot_bytes or op.slot[0] != 1) {
                    op.err = error.BadState;
                    return;
                }
                if (e.kept) |k| gpa.free(k);
                e.kept = gpa.dupe(u32, op.prompt) catch null;
                e.at = op.at;
                e.imported += 1;
                op.tokens = op.at;
            },
        }
    }
};

const Reply = struct { status: u16, head: []u8, body: []u8 };

/// One request over a socket pair, straight into the routes (the API-key gate runs before them; auth.zig tests it).
fn exchange(srv: *server.Server, path: []const u8, headers: []const u8, body: []const u8) !Reply {
    return exchangeClaiming(srv, path, headers, body.len, body, null);
}

/// As exchange, the head claiming ``claimed`` body bytes; ``unread``: bytes the server left unread (it must close).
fn exchangeClaiming(srv: *server.Server, path: []const u8, headers: []const u8, claimed: usize, body: []const u8, unread: ?*usize) !Reply {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = try std.fmt.allocPrint(gpa, "POST {s} HTTP/1.1\r\nHost: t\r\n{s}Content-Length: {d}\r\n\r\n", .{ path, headers, claimed });
    defer gpa.free(head);
    try writeAll(pair[0], head);
    try writeAll(pair[0], body);
    var conn = try Conn.init(gpa, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    if (try conn.readRequest(a, false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, a);
    if (unread) |u| {
        const left = try readReady(pair[1]);
        defer gpa.free(left);
        u.* = conn.end - conn.start + left.len;
        if (!conn.close) return error.KeptOpen;
    }
    const raw = try readReady(pair[0]);
    defer gpa.free(raw);
    const split = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.BadReply;
    const status = std.fmt.parseInt(u16, raw["HTTP/1.1 ".len..][0..3], 10) catch return error.BadReply;
    return .{ .status = status, .head = try gpa.dupe(u8, raw[0..split]), .body = try gpa.dupe(u8, raw[split + 4 ..]) };
}

fn free(r: Reply) void {
    gpa.free(r.head);
    gpa.free(r.body);
}

fn writeAll(fd: std.c.fd_t, data: []const u8) !void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = std.c.write(fd, data[sent..].ptr, data.len - sent);
        if (n <= 0) return error.Closed;
        sent += @intCast(n);
    }
}

fn readReady(fd: std.c.fd_t) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var tmp: [4096]u8 = undefined;
    while (true) {
        var pollfd = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        if (try std.posix.poll(&pollfd, 200) == 0) break;
        const n = std.posix.read(fd, &tmp) catch |e| return e;
        if (n == 0) break;
        try out.appendSlice(gpa, tmp[0..n]);
    }
    return out.toOwnedSlice(gpa);
}

fn expectStatus(r: Reply, code: u16, text: []const u8) !void {
    defer free(r);
    try std.testing.expectEqual(code, r.status);
    try std.testing.expect(std.mem.indexOf(u8, r.body, text) != null);
}

const bearer = "Authorization: Bearer sk-move\r\n";

test "the routes are absent without the key; with it they answer only to its bearer, a 401 sent with the body unread" {
    var text: DropLate = .{};
    var backend: Mover = .{};
    defer gpa.free(backend.prompt);
    var off = try server.Server.init(gpa, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer off.deinit();
    try expectStatus(try exchange(off, "/cache/probe", bearer, probe_body), 404, "unknown path");
    try expectStatus(try exchange(off, "/v1/cache/import", bearer, "junk"), 404, "unknown path");
    var problem: []const u8 = "";
    const key = try cache_key.Key.parse(key_text, &problem);
    var on = try server.Server.init(gpa, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .transfer = key }, null);
    defer on.deinit();
    try expectStatus(try exchange(on, "/cache/probe", "", probe_body), 401, "cache transfer key is required");
    try expectStatus(try exchange(on, "/cache/probe", "Authorization: Bearer sk-api\r\n", probe_body), 401, "cache transfer key is required");
    try expectStatus(try exchange(on, "/cache/import", "x-api-key: sk-api\r\n", "junk"), 401, "cache transfer key is required");
    // headers only, or a body still on its way: the 401 does not wait for it, and the connection closes
    var unread: usize = 99;
    try expectStatus(try exchangeClaiming(on, "/cache/import", "", 10 << 20, "", &unread), 401, "cache transfer key is required");
    try std.testing.expectEqual(@as(usize, 0), unread);
    try expectStatus(try exchangeClaiming(on, "/cache/import", "Authorization: Bearer not-the-bearer\r\n", 10 << 20, "0123456789", &unread), 401, "cache transfer key is required");
    try std.testing.expectEqual(@as(usize, 10), unread);
    try expectStatus(try exchangeClaiming(on, "/cache/probe", "", probe_body.len, probe_body, &unread), 401, "cache transfer key is required");
    try std.testing.expectEqual(probe_body.len, unread);
    try std.testing.expectEqual(@as(usize, 0), backend.prompt.len); // nothing reached the engine
    try expectStatus(try exchange(on, "/cache/probe", "x-cache-transfer-key: sk-move\r\n", probe_body), 200, "\"tokens\": 0");
    try expectStatus(try exchange(on, "/v1/cache/probe", "x-api-key: sk-api\r\n" ++ bearer, probe_body), 200, "\"tokens\": 0");
    try std.testing.expect(backend.prompt.len > 0);
}

test "export is a signed body import takes back; a tampered, truncated, unsigned, oversized or malformed one is refused" {
    var text: DropLate = .{};
    var backend: Mover = .{};
    defer gpa.free(backend.prompt);
    defer if (backend.kept) |k| gpa.free(k);
    var problem: []const u8 = "";
    const key = try cache_key.Key.parse(key_text, &problem);
    var srv = try server.Server.init(gpa, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .transfer = key }, null);
    defer srv.deinit();
    try expectStatus(try exchange(srv, "/cache/export", bearer, probe_body), 404, "NoState");
    // the engine keeps the probe's first 4 positions: export carries them, signed
    backend.kept = try gpa.dupe(u32, backend.prompt[0..4]);
    backend.at = 4;
    const x = try exchange(srv, "/cache/export", bearer, probe_body);
    defer free(x);
    try std.testing.expectEqual(@as(u16, 200), x.status);
    try std.testing.expect(std.mem.indexOf(u8, x.head, "x-cache-tokens: 4") != null);
    try std.testing.expect(std.mem.indexOf(u8, x.head, "Content-Type: application/octet-stream") != null);
    try std.testing.expectEqual(wire.head_bytes + 16 + 3 + 8 + wire.mac_bytes, x.body.len);
    try std.testing.expect(wire.verify(&key.mac, x.body));
    const p = try wire.parse(shape, x.body);
    try std.testing.expectEqual(@as(u32, 4), p.positions);
    // taken back, and found by the next probe
    gpa.free(backend.kept.?);
    backend.kept = null;
    try expectStatus(try exchange(srv, "/cache/probe", bearer, probe_body), 200, "\"tokens\": 0");
    try expectStatus(try exchange(srv, "/cache/import", bearer, x.body), 200, "\"tokens\": 4");
    try std.testing.expectEqual(@as(usize, 1), backend.imported);
    try expectStatus(try exchange(srv, "/cache/probe", bearer, probe_body), 200, "\"tokens\": 4");
    // a flipped payload byte, a flipped MAC byte, a cut body, another key's signature: the MAC refuses each
    const bad = try gpa.dupe(u8, x.body);
    defer gpa.free(bad);
    bad[wire.head_bytes + 2] ^= 1;
    try expectStatus(try exchange(srv, "/cache/import", bearer, bad), 400, "signature");
    @memcpy(bad, x.body);
    bad[bad.len - 1] ^= 1;
    try expectStatus(try exchange(srv, "/cache/import", bearer, bad), 400, "signature");
    try expectStatus(try exchange(srv, "/cache/import", bearer, x.body[0 .. x.body.len - 5]), 400, "signature");
    try expectStatus(try exchange(srv, "/cache/import", bearer, ""), 400, "signature");
    var other = key;
    other.mac[0] ^= 1;
    const resigned = wire.sign(&other.mac, &.{x.body[0 .. x.body.len - wire.mac_bytes]});
    @memcpy(bad, x.body);
    @memcpy(bad[bad.len - wire.mac_bytes ..], &resigned);
    try expectStatus(try exchange(srv, "/cache/import", bearer, bad), 400, "signature");
    // a signed body whose header lies is refused by the parser, naming the field; the engine saw none of these
    @memcpy(bad, x.body);
    std.mem.writeInt(u32, bad[32..36], 3, .little); // blocks: not the positions' count of them
    const lie = wire.sign(&key.mac, &.{bad[0 .. bad.len - wire.mac_bytes]});
    @memcpy(bad[bad.len - wire.mac_bytes ..], &lie);
    try expectStatus(try exchange(srv, "/cache/import", bearer, bad), 400, "blocks: not the positions' block count");
    try std.testing.expectEqual(@as(usize, 1), backend.imported);
    // longer than any state this engine takes: refused on Content-Length, the body unread; chunked: refused
    const huge = try gpa.alloc(u8, (try shape.bodyMax()) + 1);
    defer gpa.free(huge);
    @memset(huge, 0);
    try expectStatus(try exchange(srv, "/cache/import", bearer, huge), 413, "longer than any state");
    try expectStatus(try exchange(srv, "/cache/import", bearer ++ "Transfer-Encoding: chunked\r\n", "0\r\n\r\n"), 411, "Content-Length is required");
    try std.testing.expectEqual(@as(usize, 1), backend.imported);
}
