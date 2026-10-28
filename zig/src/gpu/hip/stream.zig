//! Streams and events: ordering, waits and GPU-side timing (the CUDA runtime's surface), plus a bounded wait.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const Stream = struct {
    d: *const Driver,
    handle: abi.Stream,

    /// A non-blocking stream never waits on the null stream (what every engine stream should be).
    pub fn init(d: *const Driver, non_blocking: bool) Error!Stream {
        var s: abi.Stream = null;
        try d.check(d.api.hipStreamCreateWithFlags(&s, if (non_blocking) abi.stream_non_blocking else 0), "hipStreamCreateWithFlags");
        if (s == null) return error.Invalid;
        return .{ .d = d, .handle = s };
    }

    pub fn deinit(self: *Stream) void {
        _ = self.d.api.hipStreamDestroy(self.handle);
        self.* = undefined;
    }

    pub fn synchronize(self: Stream) Error!void {
        try self.d.check(self.d.api.hipStreamSynchronize(self.handle), "hipStreamSynchronize");
    }

    /// Waits at most `ns` for the queued work: Timeout leaves the work running (the caller aborts its request and its
    /// peer); a kernel that never finishes cannot be cancelled from here.
    pub fn synchronizeFor(self: Stream, ns: u64) Error!void {
        const start = nowNs();
        var spin: u32 = 0;
        while (!try self.done()) {
            if (nowNs() - start >= ns) return error.Timeout;
            spin += 1;
            if (spin > 64) std.Thread.yield() catch {};
        }
    }

    /// True once every queued item has finished.
    pub fn done(self: Stream) Error!bool {
        self.d.check(self.d.api.hipStreamQuery(self.handle), "hipStreamQuery") catch |e| switch (e) {
            error.NotReady => return false,
            else => return e,
        };
        return true;
    }

    pub fn wait(self: Stream, event: Event) Error!void {
        try self.d.check(self.d.api.hipStreamWaitEvent(self.handle, event.handle, 0), "hipStreamWaitEvent");
    }

    /// Whether launches on this stream are being recorded into a graph.
    pub fn capturing(self: Stream) Error!bool {
        var s: abi.CaptureStatus = .none;
        try self.d.check(self.d.api.hipStreamIsCapturing(self.handle, &s), "hipStreamIsCapturing");
        return s != .none;
    }
};

pub const Event = struct {
    d: *const Driver,
    handle: abi.Event,

    /// Timing events cost a little more to record; ordering-only events skip the timestamp.
    pub fn init(d: *const Driver, timing: bool) Error!Event {
        var e: abi.Event = null;
        try d.check(d.api.hipEventCreateWithFlags(&e, if (timing) 0 else abi.event_disable_timing), "hipEventCreateWithFlags");
        if (e == null) return error.Invalid;
        return .{ .d = d, .handle = e };
    }

    pub fn deinit(self: *Event) void {
        _ = self.d.api.hipEventDestroy(self.handle);
        self.* = undefined;
    }

    pub fn record(self: Event, stream: Stream) Error!void {
        try self.d.check(self.d.api.hipEventRecord(self.handle, stream.handle), "hipEventRecord");
    }

    pub fn synchronize(self: Event) Error!void {
        try self.d.check(self.d.api.hipEventSynchronize(self.handle), "hipEventSynchronize");
    }

    pub fn done(self: Event) Error!bool {
        self.d.check(self.d.api.hipEventQuery(self.handle), "hipEventQuery") catch |e| switch (e) {
            error.NotReady => return false,
            else => return e,
        };
        return true;
    }

    /// Milliseconds between two recorded timing events.
    pub fn elapsedMs(start: Event, end: Event) Error!f32 {
        var ms: f32 = 0;
        try start.d.check(start.d.api.hipEventElapsedTime(&ms, start.handle, end.handle), "hipEventElapsedTime");
        return ms;
    }
};

/// CLOCK_MONOTONIC in nanoseconds.
pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
