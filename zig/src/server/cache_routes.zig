//! POST /cache/probe, /cache/export and /cache/import: a kept prompt state off the engine as signed bytes, and back.
const std = @import("std");
const api = @import("engine_api");
const wire = api.cache_wire;
const json = @import("json.zig");
const http_conn = @import("http_conn.zig");
const http_body = @import("http_body.zig");
const openai = @import("openai.zig");
const chat = @import("chat.zig");
const routes = @import("routes.zig");
const errors = @import("errors.zig");
const cache_key = @import("cache_key.zig");
const Server = @import("server.zig").Server;
const Conn = http_conn.Conn;
const Allocator = std.mem.Allocator;

pub const Which = api.CacheOp.Kind;

/// The three routes, with or without the /v1 prefix (the API-key gate lists them).
pub fn route(path: []const u8) ?Which {
    const p = if (std.mem.startsWith(u8, path, "/v1/")) path[3..] else path;
    if (std.mem.eql(u8, p, "/cache/probe")) return .probe;
    if (std.mem.eql(u8, p, "/cache/export")) return .export_state;
    if (std.mem.eql(u8, p, "/cache/import")) return .import_state;
    return null;
}

/// A probe or export never goes away while it is prepared.
const Stay = struct {
    pub fn check(_: Stay) bool {
        return false;
    }
};

/// Absent (404) without --cache-transfer-key; its bearer required (401) before anything of the body is read.
pub fn post(srv: *Server, conn: *Conn, a: Allocator, which: Which) void {
    const key = srv.config.transfer orelse {
        discard(conn, a);
        return routes.unknown(conn, a);
    };
    if (!key.accepts(single(conn, "Authorization"), single(conn, "x-cache-transfer-key"))) {
        discard(conn, a);
        return unauthorized(conn);
    }
    const shape = srv.engine.cacheShape() orelse {
        discard(conn, a);
        return fail(conn, a, 404, "this engine moves no prompt states");
    };
    if (which == .import_state) return importState(srv, conn, a, &key, shape);
    lookup(srv, conn, a, &key, shape, which);
}

/// probe and export: the request's messages rendered as a chat request's are, then the engine's answer for them.
fn lookup(srv: *Server, conn: *Conn, a: Allocator, key: *const cache_key.Key, shape: wire.Shape, which: Which) void {
    var cx: errors.Cx = .{ .a = a };
    const body = http_body.readJson(conn, &cx) catch return fail(conn, a, cx.status(), cx.message);
    const p = openai.plan(srv, &cx, true, body) catch return fail(conn, a, cx.status(), cx.message);
    const prepared = chat.prepare(srv, &cx, p.input, Stay{}) catch |e| return fail(conn, a, if (e == error.OutOfMemory) 500 else cx.status(), if (cx.message.len > 0) cx.message else @errorName(e));
    chat.release(srv, prepared.preparing);
    var op: api.CacheOp = .{ .kind = which, .prompt = prepared.request.prompt, .chunks = prepared.request.chunks, .gpa = srv.gpa };
    srv.engine.cacheOp(&op) catch return fail(conn, a, 404, "this engine moves no prompt states");
    if (op.err) |e| return failed(conn, a, e);
    if (which == .probe) return sendTokens(conn, a, op.tokens, prepared.request.prompt.len);
    const x = op.out.?;
    defer srv.gpa.free(x.ids);
    defer srv.gpa.free(x.parts.slot);
    defer srv.gpa.free(x.parts.rows);
    sendState(srv.gpa, conn, key, shape, x) catch {};
}

/// The export: the signed header, the ids, the slot, the blocks, the MAC; its positions in x-cache-tokens.
fn sendState(gpa: Allocator, conn: *Conn, key: *const cache_key.Key, shape: wire.Shape, x: api.prompt_cache.Exported) !void {
    const idb = try gpa.alloc(u8, 4 * x.ids.len);
    defer gpa.free(idb);
    wire.idBytes(x.ids, idb);
    const head = wire.encodeHead(shape, x.at, idb);
    const mac = wire.sign(&key.mac, &.{ &head, idb, x.parts.slot, x.parts.rows });
    const total = head.len + idb.len + x.parts.slot.len + x.parts.rows.len + mac.len;
    try conn.startResponse(200, null);
    try conn.addHeader("Content-Type", "application/octet-stream");
    var len: [24]u8 = undefined;
    try conn.addHeader("Content-Length", try std.fmt.bufPrint(&len, "{d}", .{total}));
    var tok: [16]u8 = undefined;
    try conn.addHeader("x-cache-tokens", try std.fmt.bufPrint(&tok, "{d}", .{x.at}));
    if (conn.close) try conn.addHeader("Connection", "close");
    try conn.finish(&head);
    try conn.writeAll(idb);
    try conn.writeAll(x.parts.slot);
    try conn.writeAll(x.parts.rows);
    try conn.writeAll(&mac);
}

/// import: Content-Length against the engine's largest state, one buffer of that size, the MAC, the header, the copies.
fn importState(srv: *Server, conn: *Conn, a: Allocator, key: *const cache_key.Key, shape: wire.Shape) void {
    const n = http_body.contentLength(conn) orelse {
        conn.close = true;
        return fail(conn, a, 411, "Content-Length is required: the body is a state POST /cache/export gave");
    };
    const cap = shape.bodyMax() catch return fail(conn, a, 500, "this engine's state sizes overflow");
    if (n > cap) {
        conn.close = true;
        return fail(conn, a, 413, "the body is longer than any state this engine takes");
    }
    const gpa = srv.gpa;
    var buf = std.ArrayList(u8).initCapacity(gpa, @intCast(n)) catch return fail(conn, a, 500, "out of memory");
    defer buf.deinit(gpa);
    const got = conn.readExact(gpa, &buf, @intCast(n)) catch 0;
    if (got != n) {
        conn.close = true;
        return fail(conn, a, 400, "incomplete request body");
    }
    const body = buf.items;
    if (!wire.verify(&key.mac, body)) return fail(conn, a, 400, "signature: not this server's");
    const p = wire.parse(shape, body) catch |e| return fail(conn, a, 400, wire.message(e));
    const ids = a.alloc(u32, p.idCount()) catch return fail(conn, a, 500, "out of memory");
    p.idsInto(ids);
    var op: api.CacheOp = .{ .kind = .import_state, .prompt = ids, .at = p.positions, .slot = p.slot, .rows = p.rows, .gpa = gpa };
    srv.engine.cacheOp(&op) catch return fail(conn, a, 404, "this engine moves no prompt states");
    if (op.err) |e| return failed(conn, a, e);
    sendTokens(conn, a, op.tokens, ids.len);
}

/// The engine's refusal as a status: the state (400), no room now (503), nothing to move (404), else 500.
fn failed(conn: *Conn, a: Allocator, e: anyerror) void {
    const code: u16 = switch (e) {
        error.BadState, error.NotAChunkEnd, error.PastBudget, error.StaleState, error.BadRows, error.StateTooLong => 400,
        error.ContextFull => 503,
        error.NoState, error.NoCache, error.NoTransfer => 404,
        else => 500,
    };
    fail(conn, a, code, @errorName(e));
}

fn sendTokens(conn: *Conn, a: Allocator, tokens: u32, prompt: usize) void {
    const o = json.newObject(a) catch return;
    o.put(a, "tokens", json.intValue(a, tokens) catch return) catch return;
    o.put(a, "prompt_tokens", json.intValue(a, prompt) catch return) catch return;
    routes.sendValue(conn, a, 200, .{ .object = o });
}

fn fail(conn: *Conn, a: Allocator, code: u16, message: []const u8) void {
    const inner = json.newObject(a) catch return;
    inner.put(a, "message", .{ .string = message }) catch return;
    const o = json.newObject(a) catch return;
    o.put(a, "error", .{ .object = inner }) catch return;
    routes.sendValue(conn, a, code, .{ .object = o });
}

fn unauthorized(conn: *Conn) void {
    const body = "{\"error\": {\"message\": \"the cache transfer key is required\"}}";
    conn.close = true;
    conn.startResponse(401, null) catch return;
    conn.addHeader("WWW-Authenticate", "Bearer") catch return;
    conn.addHeader("Content-Type", "application/json") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch unreachable) catch return;
    conn.addHeader("Connection", "close") catch return;
    conn.finish(body) catch {};
}

/// A refused request's body is read and dropped, so it cannot reach the next request on this connection.
fn discard(conn: *Conn, a: Allocator) void {
    _ = http_body.read(conn, a, http_body.limit) catch {};
}

/// A header that appears exactly once; none or several read as absent (as the key gate reads them).
fn single(conn: *const Conn, name: []const u8) ?[]const u8 {
    if (conn.headerCount(name) != 1) return null;
    return conn.header(name);
}
