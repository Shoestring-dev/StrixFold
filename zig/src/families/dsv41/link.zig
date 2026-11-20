//! Rank 0 to rank 1 over TCP (the fabric addresses): length-prefixed frames for each round's step and NCCL's id, as
//! multi.py's Link sends them. One connection, TCP_NODELAY, SO_BUSY_POLL, blocking reads (recvSpin polls first).
const std = @import("std");
const c = std.c;

pub const Link = struct {
    fd: c.fd_t,

    fn socket() !c.fd_t {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        tune(fd);
        return fd;
    }

    /// TCP_NODELAY, and SO_BUSY_POLL: a read polls the NIC's receive queue itself (up to 50 us) instead of waiting for
    /// its interrupt, whose core may be leaving a deep idle state (rank 0 to rank 1, 64-byte frames: round trip 1.0 ms
    /// blocking, 0.72 ms polling the socket, 11 us with SO_BUSY_POLL).
    fn tune(fd: c.fd_t) void {
        const one: c_int = 1;
        const busy_us: c_int = 50;
        _ = c.setsockopt(fd, c.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&one), @sizeOf(c_int));
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.BUSY_POLL, std.mem.asBytes(&busy_us), @sizeOf(c_int));
        // a peer gone without a FIN (power, hung kernel, cable) ends the idle read with PeerDown in 60 s, not never
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.KEEPALIVE, std.mem.asBytes(&one), @sizeOf(c_int));
        _ = c.setsockopt(fd, c.IPPROTO.TCP, std.posix.TCP.KEEPIDLE, std.mem.asBytes(&keep_idle_s), @sizeOf(c_int));
        _ = c.setsockopt(fd, c.IPPROTO.TCP, std.posix.TCP.KEEPINTVL, std.mem.asBytes(&keep_interval_s), @sizeOf(c_int));
        _ = c.setsockopt(fd, c.IPPROTO.TCP, std.posix.TCP.KEEPCNT, std.mem.asBytes(&keep_probes), @sizeOf(c_int));
    }

    /// An idle link's lost-peer deadline: a probe after keep_idle_s, then keep_probes probes keep_interval_s apart.
    pub const keep_idle_s: c_int = 30;
    pub const keep_interval_s: c_int = 10;
    pub const keep_probes: c_int = 3;

    fn addr(ip: [4]u8, port: u16) c.sockaddr.in {
        return .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip) };
    }

    /// Rank 0: listen on `ip`:`port` and take the one peer's connection.
    /// Rank 0: the follower's connection, waiting at most `seconds` for it (a follower that died while loading
    /// never connects: rank 0 then fails its open instead of waiting forever, U47).
    pub fn listen(ip: [4]u8, port: u16, seconds: u32) !Link {
        const fd = try socket();
        errdefer _ = c.close(fd);
        const one: c_int = 1;
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, std.mem.asBytes(&one), @sizeOf(c_int));
        var a = addr(ip, port);
        if (c.bind(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) != 0) return error.BindFailed;
        if (c.listen(fd, 1) != 0) return error.ListenFailed;
        var pfd = [1]c.pollfd{.{ .fd = fd, .events = c.POLL.IN, .revents = 0 }};
        while (true) {
            const r = c.poll(&pfd, 1, @intCast(@as(u64, seconds) * 1000));
            if (r > 0) break;
            if (r == 0) return error.PeerNeverLinked;
            if (c.errno(r) != .INTR) return error.AcceptFailed;
        }
        const peer = c.accept(fd, null, null);
        if (peer < 0) return error.AcceptFailed;
        _ = c.close(fd);
        tune(peer);
        return .{ .fd = peer };
    }

    /// Rank 1: connect to rank 0, retrying for up to `seconds` while it starts.
    pub fn connect(io: std.Io, ip: [4]u8, port: u16, seconds: u32) !Link {
        var tries: u32 = 0;
        while (true) : (tries += 1) {
            const fd = try socket();
            var a = addr(ip, port);
            if (c.connect(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) == 0) return .{ .fd = fd };
            _ = c.close(fd);
            if (tries >= seconds * 4) return error.ConnectFailed;
            std.Io.sleep(io, .fromMilliseconds(250), .awake) catch {};
        }
    }

    pub fn close(l: *Link) void {
        _ = c.close(l.fd);
        l.* = undefined;
    }

    fn writeAll(l: Link, bytes: []const u8) !void {
        var done: usize = 0;
        while (done < bytes.len) {
            const n = c.send(l.fd, bytes.ptr + done, bytes.len - done, 0);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    fn readAll(l: Link, out: []u8) !void {
        var done: usize = 0;
        while (done < out.len) {
            const n = c.recv(l.fd, out.ptr + done, out.len - done, 0);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    /// One frame: its length (u32, little-endian), then its bytes.
    pub fn send(l: Link, bytes: []const u8) !void {
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(bytes.len), .little);
        try l.writeAll(&len);
        try l.writeAll(bytes);
    }

    /// recv, polling the socket without blocking for up to `spin_ns` first: a frame that comes within it is read
    /// without the receiving thread's sleep and wake-up (a core's deepest idle state takes 433 us to leave on GB10; the
    /// follower's frames come a fraction of a millisecond after its GPU goes idle), and after it the read blocks.
    pub fn recvSpin(l: Link, buf: []u8, spin_ns: u64) ![]u8 {
        var len: [4]u8 = undefined;
        var got: usize = 0;
        const t0 = monotonicNs();
        var polls: u32 = 0;
        while (got < len.len) {
            const n = c.recv(l.fd, @as([*]u8, &len) + got, len.len - got, c.MSG.DONTWAIT);
            if (n > 0) {
                got += @intCast(n);
                continue;
            }
            if (n == 0) return error.PeerDown;
            switch (c.errno(n)) {
                .AGAIN => {},
                .INTR => continue,
                else => return error.PeerDown,
            }
            polls +%= 1;
            if (polls % 64 == 0 and monotonicNs() -% t0 > spin_ns) break;
            std.atomic.spinLoopHint();
        }
        if (got < len.len) try l.readAll(len[got..]);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }

    fn monotonicNs() u64 {
        var ts: c.timespec = undefined;
        _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    /// The next frame into `buf` (a frame longer than `buf` is refused); its bytes.
    pub fn recv(l: Link, buf: []u8) ![]u8 {
        var len: [4]u8 = undefined;
        try l.readAll(&len);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }
};

/// "192.0.2.7" as four bytes.
pub fn parseIp(text: []const u8) ![4]u8 {
    var out: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, text, '.');
    for (&out) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
    if (it.next() != null) return error.BadAddress;
    return out;
}

test "addresses parse as four bytes" {
    try std.testing.expectEqual([4]u8{ 192, 0, 2, 7 }, try parseIp("192.0.2.7"));
    try std.testing.expectError(error.BadAddress, parseIp("192.0.2"));
}

test "frames cross a loopback link in order" {
    const io = std.testing.io;
    const port: u16 = 29000 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 500)));
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            defer l.close();
            l.send("step 1") catch return;
            l.send(&.{}) catch return;
            l.send("polled") catch return;
            l.send("late") catch return;
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port, 10);
    defer l.close();
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("step 1", try l.recv(&buf));
    try std.testing.expectEqual(@as(usize, 0), (try l.recv(&buf)).len);
    try std.testing.expectEqualStrings("polled", try l.recvSpin(&buf, 1000 * std.time.ns_per_ms));
    try std.testing.expectEqualStrings("late", try l.recvSpin(&buf, 0));
    t.join();
}

test "both ends of a link probe an idle peer: a lost one ends the wait within 60 s" {
    const port: u16 = 30000 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds, 400)));
    const Peer = struct {
        var fd: c.fd_t = -1;
        fn run(p: u16) void {
            const l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            fd = l.fd;
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port, 10);
    defer l.close();
    t.join();
    try std.testing.expect(Peer.fd >= 0);
    defer _ = c.close(Peer.fd);
    for ([_]c.fd_t{ l.fd, Peer.fd }) |fd| {
        const Opt = struct { level: i32, name: u32, want: c_int };
        for ([_]Opt{
            .{ .level = c.SOL.SOCKET, .name = c.SO.KEEPALIVE, .want = 1 },
            .{ .level = c.IPPROTO.TCP, .name = std.posix.TCP.KEEPIDLE, .want = Link.keep_idle_s },
            .{ .level = c.IPPROTO.TCP, .name = std.posix.TCP.KEEPINTVL, .want = Link.keep_interval_s },
            .{ .level = c.IPPROTO.TCP, .name = std.posix.TCP.KEEPCNT, .want = Link.keep_probes },
        }) |o| {
            var got: c_int = -1;
            var len: c.socklen_t = @sizeOf(c_int);
            try std.testing.expectEqual(@as(c_int, 0), c.getsockopt(fd, o.level, o.name, std.mem.asBytes(&got), &len));
            try std.testing.expectEqual(o.want, got);
        }
    }
    try std.testing.expect(Link.keep_idle_s + Link.keep_probes * Link.keep_interval_s <= 60);
}

test "rank 0 stops waiting for a follower that never connects" {
    const port: u16 = 29500 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds, 400)));
    try std.testing.expectError(error.PeerNeverLinked, Link.listen(.{ 127, 0, 0, 1 }, port, 1));
}
