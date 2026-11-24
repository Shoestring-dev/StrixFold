//! /health answers 503 while the engine's status says failed (a TP pair that broke), and 200 otherwise.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;

/// Byte tokens and a one-line prompt.
const ByteText = struct {
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
        return a.dupe(u8, "hi\n");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An engine whose status says failed or not.
const Failing = struct {
    failed: bool = false,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Busy;
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const e: *Failing = @ptrCast(@alignCast(ctx));
        out.* = .{ .failed = e.failed };
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The whole HTTP reply to GET /health.
fn health(a: std.mem.Allocator, srv: *server.Server) ![]u8 {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = "GET /health HTTP/1.1\r\nHost: t\r\n\r\n";
    if (std.c.write(pair[0], head.ptr, head.len) != head.len) return error.Closed;
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    if (try conn.readRequest(arena.allocator(), false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, arena.allocator());
    var got: std.ArrayList(u8) = .empty;
    var tmp: [4096]u8 = undefined;
    var pollfd = [_]std.posix.pollfd{.{ .fd = pair[0], .events = std.posix.POLL.IN, .revents = 0 }};
    while (try std.posix.poll(&pollfd, 200) > 0) {
        const n = try std.posix.read(pair[0], &tmp);
        if (n == 0) break;
        try got.appendSlice(a, tmp[0..n]);
    }
    return got.items;
}

test "/health answers 503 while the engine's status says failed, and 200 again otherwise" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: ByteText = .{};
    var backend: Failing = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
    defer srv.deinit();
    try std.testing.expect(std.mem.startsWith(u8, try health(a, srv), "HTTP/1.1 200"));
    backend.failed = true;
    const reply = try health(a, srv);
    try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 503"));
    try std.testing.expect(std.mem.indexOf(u8, reply, "\"failed\"") != null);
    backend.failed = false;
    try std.testing.expect(std.mem.startsWith(u8, try health(a, srv), "HTTP/1.1 200"));
}
