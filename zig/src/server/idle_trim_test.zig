//! Conn.trimIdle: between requests a reply buffer grown past the limit goes back, and a small one stays.
const std = @import("std");
const Conn = @import("http_conn.zig").Conn;

test "an idle connection gives back a reply buffer grown past the limit and keeps a small one" {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    try conn.out.ensureTotalCapacity(conn.gpa, 1 << 20); // a long reply's buffer
    conn.trimIdle(64 << 10);
    try std.testing.expectEqual(@as(usize, 0), conn.out.capacity);
    try conn.out.ensureTotalCapacity(conn.gpa, 4 << 10);
    const kept = conn.out.capacity;
    conn.trimIdle(64 << 10);
    try std.testing.expectEqual(kept, conn.out.capacity);
}
