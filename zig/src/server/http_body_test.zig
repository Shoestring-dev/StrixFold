//! The request body limit: 256 MiB, so a request of many full-size base64 pictures is read, one byte over is refused.
const std = @import("std");
const http_body = @import("http_body.zig");
const Conn = @import("http_conn.zig").Conn;
const gpa = std.testing.allocator;

const mib = 1024 * 1024;

/// The framing answer for a request that declares `length` body bytes and then closes without sending any.
fn declared(length: usize) !http_body.Body {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[1]);
    const request = "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n";
    const head = try std.fmt.allocPrint(gpa, request, .{length});
    defer gpa.free(head);
    if (std.c.write(pair[0], head.ptr, head.len) != @as(isize, @intCast(head.len))) return error.Closed;
    _ = std.c.close(pair[0]); // the body never comes: inside the limit reads as incomplete, above it is refused first
    var conn = try Conn.init(gpa, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    if (try conn.readRequest(a, false) != .ready) return error.BadRequest;
    const body = try http_body.read(&conn, a, http_body.limit);
    return switch (body) {
        .ok => .{ .ok = "" },
        .refused => |m| .{ .refused = try gpa.dupe(u8, m) },
    };
}

test "request body limit is 256 MiB: a length up to it is read, one byte over is refused naming it" {
    try std.testing.expectEqual(@as(usize, 256 * mib), http_body.limit);
    for ([_]usize{ 33 * mib, 200 * mib, 256 * mib }) |n| { // inside the limit: read, not refused as too big
        const r = try declared(n);
        defer if (r == .refused) gpa.free(r.refused);
        try std.testing.expectEqualStrings("incomplete request body", r.refused);
    }
    const over = try declared(256 * mib + 1);
    defer gpa.free(over.refused);
    try std.testing.expectEqualStrings("request body exceeds the 256 MiB limit", over.refused);
}
