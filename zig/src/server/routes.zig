//! Method and path dispatch, as ``do_GET``, ``do_POST`` and ``do_DELETE`` route them.
const std = @import("std");
const json = @import("json.zig");
const http_conn = @import("http_conn.zig");
const http_body = @import("http_body.zig");
const auth = @import("auth.zig");
const openai = @import("openai.zig");
const responses = @import("responses.zig");
const anthropic = @import("anthropic.zig");
const tokens = @import("tokens.zig");
const decisions = @import("decisions.zig");
const status = @import("status_routes.zig");
const queue_cap = @import("queue_cap.zig");
const Server = @import("server.zig").Server;
const Conn = http_conn.Conn;
const Allocator = std.mem.Allocator;

pub fn dispatch(srv: *Server, conn: *Conn, a: Allocator) void {
    const m = conn.method;
    if (std.mem.eql(u8, m, "GET")) return get(srv, conn, a);
    if (std.mem.eql(u8, m, "POST")) return post(srv, conn, a);
    if (std.mem.eql(u8, m, "DELETE")) return responses.delete(srv, conn, a, responses.route(auth.routePath(conn.path)));
    const message = std.fmt.allocPrint(a, "Unsupported method ({s})", .{http_conn.pyRepr(a, m) catch m}) catch return;
    conn.sendError(a, 501, message, null) catch {};
}

/// Python's ``json.dumps(payload)`` sent as a JSON reply.
pub fn sendValue(conn: *Conn, a: Allocator, code: u16, payload: json.Value) void {
    const body = json.stringify(a, payload, .{}) catch return;
    conn.sendJson(code, body);
}

pub fn unknown(conn: *Conn, a: Allocator) void {
    const message = std.fmt.allocPrint(a, "unknown path {s}", .{conn.requestPath(a)}) catch return;
    const body = std.fmt.allocPrint(a, "{{\"error\": {{\"message\": {s}}}}}", .{json.quote(a, message, .{}) catch return}) catch return;
    conn.sendJson(404, body);
}

fn get(srv: *Server, conn: *Conn, a: Allocator) void {
    const route = auth.routePath(conn.path);
    if (responses.route(route)) |rid| if (rid.len > 0) return responses.get(srv, conn, a, rid);
    if (std.mem.eql(u8, route, "/metrics") or std.mem.eql(u8, route, "/v1/metrics")) return status.metrics(srv, conn, a);
    if (std.mem.eql(u8, route, "/dashboard")) return status.dashboard(srv, conn, a);
    if (std.mem.eql(u8, route, "/stats")) return status.stats(srv, conn, a);
    if (route.len == 0 or std.mem.eql(u8, route, "/health")) return status.health(srv, conn, a) catch {};
    if (std.mem.endsWith(u8, route, "/models")) return status.models(srv, conn, a) catch {};
    unknown(conn, a);
}

/// A refused route's body is read and dropped, so it cannot reach the next request on this connection.
fn discardBody(conn: *Conn, a: Allocator) void {
    _ = http_body.read(conn, a, http_body.limit) catch {};
}

fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    const route = auth.routePath(conn.path);
    if (std.mem.endsWith(u8, route, "/decisions")) return decisions.post(srv, conn, a);
    if (anthropic.route(conn.path)) return if (std.mem.endsWith(u8, route, "/count_tokens")) anthropic.post(srv, conn, a) else gated(srv, conn, a, .messages);
    if (responses.route(route)) |rid| if (rid.len == 0) return gated(srv, conn, a, .responses);
    for (tokens.paths) |r| if (std.mem.eql(u8, route, r)) return tokens.post(srv, conn, a, std.mem.endsWith(u8, route, "/detokenize"));
    const is_chat = std.mem.endsWith(u8, route, "/chat/completions");
    if (!is_chat and !std.mem.endsWith(u8, route, "/completions")) {
        discardBody(conn, a);
        return unknown(conn, a);
    }
    gated(srv, conn, a, if (is_chat) .chat else .completions);
}

const Generation = enum { chat, completions, messages, responses };

/// The longest body a refused request has read and dropped before its 429.
const drain_max = 1 << 20;

/// A generating route behind the queue cap: its place is held to the end of its reply; when full, 429 at the head.
fn gated(srv: *Server, conn: *Conn, a: Allocator, which: Generation) void {
    var ticket: queue_cap.Ticket = undefined;
    if (!srv.queue.enter(&ticket, conn.arena)) {
        const length = if (conn.header("Content-Length")) |v| std.fmt.parseInt(u64, std.mem.trim(u8, v, " "), 10) catch null else null;
        // drained, not closed: a close over unread bytes resets the connection and the client may lose the 429
        if (length != null and length.? <= drain_max and conn.header("Transfer-Encoding") == null) discardBody(conn, a) else conn.close = true;
        const message = json.quote(a, queue_cap.fullMessage(a, &srv.queue) catch return, .{}) catch return;
        const body = if (which == .messages)
            std.fmt.allocPrint(a, "{{\"type\": \"error\", \"error\": {{\"type\": \"rate_limit_error\", \"message\": {s}}}}}", .{message}) catch return
        else
            std.fmt.allocPrint(a, "{{\"error\": {{\"message\": {s}, \"type\": \"requests\", \"param\": null, \"code\": \"rate_limit_exceeded\"}}}}", .{message}) catch return;
        return conn.sendJson(429, body);
    }
    defer ticket.leave();
    switch (which) {
        .chat, .completions => openai.post(srv, conn, a, which == .chat),
        .messages => anthropic.post(srv, conn, a),
        .responses => responses.post(srv, conn, a),
    }
}
