//! Rank 0's way out once the TP pair failed: _exit, so no atexit handler or library fini can keep the process alive.
const std = @import("std");

/// The exit status a failed pair ends rank 0 with (the launcher reads it as the pair's failure).
pub const code: u8 = 3;

extern "c" fn fflush(stream: ?*anyopaque) c_int;
extern "c" fn alarm(seconds: c_uint) c_uint;

/// Seconds the flush may take before SIGALRM's default action ends the process instead.
pub const flush_backstop_s: c_uint = 2;

/// Ends the process now with C stdio flushed, skipping exit()'s atexit handlers and library fini (they can block).
pub fn now() noreturn {
    _ = alarm(flush_backstop_s); // a thread holding a stdio lock cannot keep the flush, and so the process, alive
    _ = fflush(null);
    std.c._exit(code);
}

extern "c" fn atexit(f: *const fn () callconv(.c) void) c_int;

fn holdForever() callconv(.c) void {
    while (true) {
        const ts: std.c.timespec = .{ .sec = 60, .nsec = 0 };
        _ = std.c.nanosleep(&ts, null);
    }
}

/// `leave`'s exit status in a forked child whose atexit handler never returns; null: alive after `wait_ms` (killed).
pub fn statusUnderHeldFini(comptime leave: fn () noreturn, wait_ms: u32) !?u8 {
    const pid = std.c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        if (atexit(holdForever) != 0) std.c._exit(100);
        const quiet = std.c.open("/dev/null", .{ .ACCMODE = .WRONLY });
        if (quiet >= 0) _ = std.c.dup2(quiet, 2); // the child's log lines stay out of the test runner's output
        leave();
    }
    var waited: u32 = 0;
    while (waited < wait_ms) : (waited += 10) {
        var st: c_int = 0;
        const r = std.c.waitpid(pid, &st, std.c.W.NOHANG);
        if (r == pid) {
            const s: u32 = @bitCast(st);
            if (!std.c.W.IFEXITED(s)) return error.ChildSignalled;
            return std.c.W.EXITSTATUS(s);
        }
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 10 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
    }
    _ = std.c.kill(pid, std.c.SIG.KILL);
    var st: c_int = 0;
    _ = std.c.waitpid(pid, &st, 0);
    return null;
}

fn libcExit() noreturn {
    std.process.exit(code);
}

test "a failed pair's exit ends the process even when an atexit handler never returns" {
    try std.testing.expectEqual(@as(?u8, code), try statusUnderHeldFini(now, 5000));
}

test "libc exit stays alive under a held atexit handler (the wedge the direct exit avoids)" {
    try std.testing.expectEqual(@as(?u8, null), try statusUnderHeldFini(libcExit, 300));
}

extern "c" var stdout: ?*anyopaque;
extern "c" fn fputs(s: [*:0]const u8, stream: ?*anyopaque) c_int;

/// What a forked child that buffers one C stdio line into a pipe and then runs `leave` wrote to that pipe.
fn pipedLine(comptime leave: fn () noreturn, buf: []u8) ![]const u8 {
    var fds: [2]std.c.fd_t = undefined;
    if (std.c.pipe(&fds) != 0) return error.PipeFailed;
    const pid = std.c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        _ = std.c.close(fds[0]);
        if (std.c.dup2(fds[1], 1) < 0) std.c._exit(100);
        _ = fputs("the last line\n", stdout); // (a pipe: fully buffered, so only a flush writes it)
        leave();
    }
    _ = std.c.close(fds[1]);
    defer _ = std.c.close(fds[0]);
    var n: usize = 0;
    while (n < buf.len) {
        const r = std.c.read(fds[0], buf[n..].ptr, buf.len - n);
        if (r <= 0) break;
        n += @intCast(r);
    }
    var st: c_int = 0;
    _ = std.c.waitpid(pid, &st, 0);
    return buf[0..n];
}

fn bareExit() noreturn {
    std.c._exit(code);
}

test "a failed pair's exit flushes C stdio first, so its last log line is not lost" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("the last line\n", try pipedLine(now, &buf));
}

test "a bare _exit drops a buffered C stdio line (what the flush is for)" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("", try pipedLine(bareExit, &buf));
}

extern "c" fn flockfile(stream: ?*anyopaque) void;

fn holdStdout(locked: *std.atomic.Value(bool)) void {
    flockfile(stdout);
    locked.store(true, .release);
    while (true) {
        const ts: std.c.timespec = .{ .sec = 60, .nsec = 0 };
        _ = std.c.nanosleep(&ts, null);
    }
}

/// The signal that ends a forked child running `now` while another thread holds stdout's lock; null: alive (killed).
fn signalUnderHeldStdio(wait_ms: u32) !?u32 {
    const pid = std.c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        var locked = std.atomic.Value(bool).init(false);
        _ = std.Thread.spawn(.{}, holdStdout, .{&locked}) catch std.c._exit(100);
        while (!locked.load(.acquire)) std.Thread.yield() catch {};
        now();
    }
    var waited: u32 = 0;
    while (waited < wait_ms) : (waited += 10) {
        var st: c_int = 0;
        if (std.c.waitpid(pid, &st, std.c.W.NOHANG) == pid) {
            const s: u32 = @bitCast(st);
            if (!std.c.W.IFSIGNALED(s)) return error.ChildExited;
            return @backingInt(std.c.W.TERMSIG(s));
        }
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 10 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
    }
    _ = std.c.kill(pid, std.c.SIG.KILL);
    var st: c_int = 0;
    _ = std.c.waitpid(pid, &st, 0);
    return null;
}

test "a failed pair's exit still ends the process when another thread holds a stdio lock through the flush" {
    const wait_ms: u32 = (flush_backstop_s + 3) * 1000;
    try std.testing.expectEqual(@as(?u32, @backingInt(std.c.SIG.ALRM)), try signalUnderHeldStdio(wait_ms));
}
