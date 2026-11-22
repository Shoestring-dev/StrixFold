//! Rank 0's way out once the TP pair failed: _exit, so no atexit handler or library fini can keep the process alive.
const std = @import("std");

/// The exit status a failed pair ends rank 0 with (the launcher reads it as the pair's failure).
pub const code: u8 = 3;

/// Ends the process now, skipping libc exit()'s atexit handlers and library fini (the GPU runtime's can block).
pub fn now() noreturn {
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
fn statusUnderHeldFini(comptime leave: fn () noreturn, wait_ms: u32) !?u8 {
    const pid = std.c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        if (atexit(holdForever) != 0) std.c._exit(100);
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
