//! Cancellable encoder pipes with one deadline covering startup, input transfer and output.
const std = @import("std");
const CancelCheck = @import("lanes").stream.CancelCheck;

pub const Transfer = struct {
    io: std.Io,
    cancel: ?CancelCheck,
    deadline: i96,

    pub fn init(io: std.Io, cancel: ?CancelCheck, seconds: u32) Transfer {
        return .{ .io = io, .cancel = cancel,
            .deadline = std.Io.Clock.awake.now(io).toNanoseconds() + @as(i96, seconds) * std.time.ns_per_s };
    }

    pub fn ready(t: *const Transfer, fd: std.c.fd_t, events: i16) !void {
        var fds = [_]std.c.pollfd{.{ .fd = fd, .events = events, .revents = 0 }};
        while (true) {
            if (t.cancel) |c| if (c.check(c.ptr)) return error.Cancelled;
            const left = t.deadline - std.Io.Clock.awake.now(t.io).toNanoseconds();
            if (left <= 0) return error.VisionCpuTimeout;
            const ms: c_int = @intCast(@min(100, @divTrunc(left + std.time.ns_per_ms - 1, std.time.ns_per_ms)));
            const count = std.c.poll(&fds, 1, ms);
            if (count < 0) {
                if (std.c.errno(count) == .INTR) continue;
                return error.VisionCpuPipe;
            }
            if (fds[0].revents & std.c.POLL.NVAL != 0) return error.VisionCpuPipe;
            if (events == std.c.POLL.OUT and fds[0].revents & (std.c.POLL.ERR | std.c.POLL.HUP) != 0)
                return error.VisionCpuPipe;
            if (count > 0) return;
        }
    }

    pub fn write(t: *const Transfer, fd: std.c.fd_t, bytes: []const u8) !void {
        const flags = std.c.fcntl(fd, std.c.F.GETFL);
        const nonblocking: c_int = @bitCast(std.c.O{ .NONBLOCK = true });
        if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblocking) < 0) return error.VisionCpuPipe;
        var at: usize = 0;
        while (at < bytes.len) {
            try t.ready(fd, std.c.POLL.OUT);
            const n = std.c.write(fd, bytes[at..].ptr, bytes.len - at);
            if (n > 0) {
                at += @intCast(n);
            } else if (n == 0 or (std.c.errno(n) != .AGAIN and std.c.errno(n) != .INTR)) {
                return error.VisionCpuPipe;
            }
        }
    }

    pub fn read(t: *const Transfer, fd: std.c.fd_t, bytes: []u8) !void {
        var at: usize = 0;
        while (at < bytes.len) {
            try t.ready(fd, std.c.POLL.IN);
            const n = std.c.read(fd, bytes[at..].ptr, bytes.len - at);
            if (n > 0) {
                at += @intCast(n);
            } else if (n == 0 or (std.c.errno(n) != .AGAIN and std.c.errno(n) != .INTR)) {
                return error.VisionCpuPipe;
            }
        }
    }
};

test "a stalled encoder input expires and its child is reaped" {
    const io = std.testing.io;
    var child = try std.process.spawn(io, .{ .argv = &.{ "/bin/sh", "-c", "exec sleep 30" }, .stdin = .pipe });
    defer child.kill(io);
    const payload = try std.testing.allocator.alloc(u8, 1 << 20);
    defer std.testing.allocator.free(payload);
    @memset(payload, 0);
    var transfer = Transfer.init(io, null, 1);
    transfer.deadline = std.Io.Clock.awake.now(io).toNanoseconds() + 50 * std.time.ns_per_ms;
    try std.testing.expectError(error.VisionCpuTimeout, transfer.write(child.stdin.?.handle, payload));
    const pid = child.id.?;
    child.kill(io);
    try std.testing.expect(child.id == null);
    var status: c_int = 0;
    const result = std.c.waitpid(pid, &status, std.c.W.NOHANG);
    try std.testing.expect(result < 0 and std.c.errno(result) == .CHILD);
}

test "a full input pipe observes cancellation" {
    var fds: [2]std.c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.pipe(&fds));
    defer _ = std.c.close(fds[0]);
    defer _ = std.c.close(fds[1]);
    const Check = struct {
        calls: usize = 0,
        fn cancelled(ptr: *anyopaque) bool {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            return self.calls >= 3;
        }
    };
    var check: Check = .{};
    const transfer = Transfer.init(std.testing.io, .{ .ptr = &check, .check = Check.cancelled }, 2);
    const payload = try std.testing.allocator.alloc(u8, 1 << 20);
    defer std.testing.allocator.free(payload);
    @memset(payload, 0);
    try std.testing.expectError(error.Cancelled, transfer.write(fds[1], payload));
}

test "a closed encoder input fails without blocking" {
    var fds: [2]std.c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.pipe(&fds));
    _ = std.c.close(fds[0]);
    defer _ = std.c.close(fds[1]);
    const transfer = Transfer.init(std.testing.io, null, 1);
    try std.testing.expectError(error.VisionCpuPipe, transfer.write(fds[1], "input"));
}

test "output waiting observes cancellation and an invalid pipe" {
    var marker: u8 = 0;
    const check = struct {
        fn cancelled(_: *anyopaque) bool { return true; }
    }.cancelled;
    const cancelled = Transfer.init(std.testing.io, .{ .ptr = &marker, .check = check }, 1);
    try std.testing.expectError(error.Cancelled, cancelled.ready(-1, std.c.POLL.IN));
    const transfer = Transfer.init(std.testing.io, null, 1);
    try std.testing.expectError(error.VisionCpuPipe, transfer.ready(1000000, std.c.POLL.IN));
}
