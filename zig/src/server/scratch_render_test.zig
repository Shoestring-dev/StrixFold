//! chat.prepare renders in a scratch arena: a request then holds its body, parse and ids, and its ids are the same.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;

/// Bytes as tokens; every prompt renders as "prompt", leaving ``waste`` bytes in its allocator as a template does.
const WasteText = struct {
    waste: usize = 0,

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

    fn render(ctx: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        const t: *WasteText = @ptrCast(@alignCast(ctx));
        if (t.waste > 0) @memset(try a.alloc(u8, t.waste), 1);
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};
/// The arena the test's request is read into, which the engine measures at submit.
var request_arena: ?*std.heap.ArenaAllocator = null;

/// One lane that answers "ok" at once; at submit it notes the request arena's capacity and the prompt ids.
const OkEngine = struct {
    capacity_at_submit: usize = 0,
    prompt: [64]u32 = undefined,
    prompt_len: usize = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *OkEngine = @ptrCast(@alignCast(ctx));
        if (request_arena) |ar| e.capacity_at_submit = ar.queryCapacity();
        e.prompt_len = @min(req.prompt.len, e.prompt.len);
        @memcpy(e.prompt[0..e.prompt_len], req.prompt[0..e.prompt_len]);
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

/// The status code of one chat request read into its own arena, as a connection reads it, over a socket pair.
fn postChat(srv: *server.Server) ![3]u8 {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    var head_buf: [256]u8 = undefined;
    const head = try std.fmt.bufPrint(&head_buf, "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{chat_body.len});
    for ([_][]const u8{ head, chat_body }) |part| if (std.c.write(pair[0], part.ptr, part.len) != part.len) return error.Closed;
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    request_arena = &arena;
    defer request_arena = null;
    if (try conn.readRequest(arena.allocator(), false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, arena.allocator());
    var tmp: [4096]u8 = undefined;
    const n = std.posix.read(pair[0], &tmp) catch 0;
    if (n < 12) return error.NoReply;
    return tmp[9..12].*;
}

fn start(text: *WasteText, backend: *OkEngine) !*server.Server {
    return server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false, .enable_thinking = false }, null);
}

test "the renders' garbage is freed before the request is submitted: its arena holds its body, parse and ids" {
    var text: WasteText = .{ .waste = 8 << 20 }; // every render leaves 8 MiB, as a long prompt's template can
    var backend: OkEngine = .{};
    const srv = try start(&text, &backend);
    defer srv.deinit();
    try std.testing.expectEqualStrings("200", &try postChat(srv));
    try std.testing.expect(backend.capacity_at_submit > chat_body.len);
    try std.testing.expect(backend.capacity_at_submit < 1 << 20);
}

test "the ids submitted are the render's, with or without garbage in the renders" {
    for ([_]usize{ 0, 8 << 20 }) |waste| {
        var text: WasteText = .{ .waste = waste };
        var backend: OkEngine = .{};
        const srv = try start(&text, &backend);
        defer srv.deinit();
        try std.testing.expectEqualStrings("200", &try postChat(srv));
        const want = "prompt";
        try std.testing.expectEqual(want.len, backend.prompt_len);
        for (want, backend.prompt[0..backend.prompt_len]) |byte, id| try std.testing.expectEqual(@as(u32, byte), id);
    }
}
