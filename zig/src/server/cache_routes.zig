//! POST /cache/probe, /cache/export, /cache/import: a conversation's kept prompt state off the engine and back later.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const http_conn = @import("http_conn.zig");
const http_body = @import("http_body.zig");
const openai = @import("openai.zig");
const chat = @import("chat.zig");
const routes = @import("routes.zig");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const Conn = http_conn.Conn;
const Allocator = std.mem.Allocator;

/// An imported state's bytes at most (a DeepSeek-V4.1 state at 512k positions is about half a GiB).
const import_limit: usize = 4 << 30;

pub fn route(path: []const u8) ?Which {
    if (std.mem.eql(u8, path, "/cache/probe")) return .probe;
    if (std.mem.eql(u8, path, "/cache/export")) return .export_state;
    if (std.mem.eql(u8, path, "/cache/import")) return .import_state;
    return null;
}

pub const Which = api.CacheOp.Kind;

/// A turn's request that never goes away: probe and export read the prompt and return.
const Stay = struct {
    pub fn check(_: Stay) bool {
        return false;
    }
};

pub fn post(srv: *Server, conn: *Conn, a: Allocator, which: Which) void {
    if (which == .import_state) return importState(srv, conn, a);
    var cx: errors.Cx = .{ .a = a };
    const body = http_body.readJson(conn, &cx) catch return fail(conn, a, cx.status(), cx.message);
    const p = openai.plan(srv, &cx, true, body) catch return fail(conn, a, cx.status(), cx.message);
    const prepared = chat.prepare(srv, &cx, p.input, Stay{}) catch return fail(conn, a, cx.status(), cx.message);
    chat.release(srv, prepared.preparing);
    var op: api.CacheOp = .{ .kind = which, .prompt = prepared.request.prompt, .chunks = prepared.request.chunks, .gpa = a };
    srv.engine.cacheOp(&op) catch return fail(conn, a, 404, "this engine keeps no prompt states to move");
    if (op.err) |e| return fail(conn, a, if (e == error.NoState or e == error.NoCache) 404 else 500, @errorName(e));
    if (which == .probe) return sendTokens(conn, a, op.tokens);
    sendState(conn, op.out, op.tokens);
}

fn importState(srv: *Server, conn: *Conn, a: Allocator) void {
    const body = switch (http_body.read(conn, a, import_limit) catch return) {
        .refused => |m| return fail(conn, a, 400, m),
        .ok => |b| b,
    };
    if (body.len == 0) return fail(conn, a, 400, "the body is a state POST /cache/export gave");
    var op: api.CacheOp = .{ .kind = .import_state, .bytes = body, .gpa = a };
    srv.engine.cacheOp(&op) catch return fail(conn, a, 404, "this engine keeps no prompt states to move");
    if (op.err) |e| {
        const refused = e == error.BadState or e == error.OtherIdentity or e == error.StateFileMismatch or e == error.StateFileVersion or
            e == error.StateFileShort or e == error.StateFileCorrupt or e == error.NotAChunkEnd or e == error.PastBudget;
        return fail(conn, a, if (refused) 400 else if (e == error.ContextFull) 503 else 500, @errorName(e));
    }
    sendTokens(conn, a, op.tokens);
}

fn sendTokens(conn: *Conn, a: Allocator, tokens: u32) void {
    const o = json.newObject(a) catch return;
    o.put(a, "tokens", json.intValue(a, tokens) catch return) catch return;
    routes.sendValue(conn, a, 200, .{ .object = o });
}

/// The state's bytes, with the positions it covers in x-cache-tokens.
fn sendState(conn: *Conn, bytes: []const u8, tokens: u32) void {
    sendStateOr(conn, bytes, tokens) catch {};
}

fn sendStateOr(conn: *Conn, bytes: []const u8, tokens: u32) !void {
    try conn.startResponse(200, null);
    try conn.addHeader("Content-Type", "application/octet-stream");
    var len: [24]u8 = undefined;
    try conn.addHeader("Content-Length", try std.fmt.bufPrint(&len, "{d}", .{bytes.len}));
    var tok: [16]u8 = undefined;
    try conn.addHeader("x-cache-tokens", try std.fmt.bufPrint(&tok, "{d}", .{tokens}));
    if (conn.close) try conn.addHeader("Connection", "close");
    try conn.finish(bytes);
}

fn fail(conn: *Conn, a: Allocator, code: u16, message: []const u8) void {
    const inner = json.newObject(a) catch return;
    inner.put(a, "message", .{ .string = message }) catch return;
    const o = json.newObject(a) catch return;
    o.put(a, "error", .{ .object = inner }) catch return;
    routes.sendValue(conn, a, code, .{ .object = o });
}
