//! Host bytes to the device in a stream's order, through a pinned ring: the caller's bytes are copied before upload
//! returns, so it may pass stack arrays and reuse them at once (the rule CUDA documents for pageable copies and the
//! dsv41 family relies on, prompt.zig `upload`; HIP does not document it). A ring region is reused only after the
//! event recorded behind its copy has completed. One Staging serves one thread; uploads during stream capture are
//! refused (a captured copy would bake the ring's address and replay stale bytes).

const std = @import("std");

/// The ring for backend `G` (the HIP runtime: its Driver, HostBuffer, Event, Stream and abi).
pub fn Staging(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const max_inflight = 128;
        const align_bytes = 256;

        d: *const G.Driver,
        ring: G.HostBuffer,
        events: [max_inflight]G.Event,
        /// In-flight regions in submission order: [start, end) of the ring and the event behind their copy.
        q_start: [max_inflight]usize = undefined,
        q_end: [max_inflight]usize = undefined,
        q_event: [max_inflight]usize = undefined,
        q_len: usize = 0,
        free_events: [max_inflight]usize = undefined,
        free_len: usize = 0,
        head: usize = 0,
        /// Times an upload waited for a region still in flight, and bytes staged (telemetry, tests).
        waits: u64 = 0,
        staged: u64 = 0,

        pub fn init(d: *const G.Driver, bytes: usize) G.Error!Self {
            if (bytes < 2 * align_bytes) return error.Invalid;
            var s: Self = .{ .d = d, .ring = try G.HostBuffer.allocFlags(d, bytes, .{}), .events = undefined };
            errdefer s.ring.free();
            var made: usize = 0;
            errdefer for (s.events[0..made]) |*e| e.deinit();
            for (&s.events, 0..) |*e, i| {
                e.* = try G.Event.init(d, false);
                made += 1;
                s.free_events[i] = max_inflight - 1 - i;
            }
            s.free_len = max_inflight;
            return s;
        }

        /// Waits for every staged copy, then frees the ring and its events.
        pub fn deinit(s: *Self) void {
            for (s.q_event[0..s.q_len]) |ev| s.events[ev].synchronize() catch {};
            for (&s.events) |*e| e.deinit();
            s.ring.free();
            s.* = undefined;
        }

        /// `bytes` to device address `dst` after the work already queued on `stream`; `bytes` may change once this
        /// returns. Uploads larger than half the ring go in half-ring pieces, in order.
        pub fn upload(s: *Self, stream: G.Stream, dst: u64, bytes: []const u8) G.Error!void {
            if (bytes.len == 0) return;
            if (try G.streamCapturing(stream)) return error.InCapture;
            const piece = s.ring.bytes.len / 2;
            var off: usize = 0;
            while (off < bytes.len) {
                const n = @min(piece, bytes.len - off);
                const at = try s.reserve(n);
                @memcpy(s.ring.bytes[at..][0..n], bytes[off..][0..n]);
                try G.uploadPinned(s.d, stream, dst + off, s.ring.bytes[at..][0..n]);
                const ev = s.free_events[s.free_len - 1];
                try s.events[ev].record(stream);
                s.free_len -= 1;
                s.q_start[s.q_len] = at;
                s.q_end[s.q_len] = at + n;
                s.q_event[s.q_len] = ev;
                s.q_len += 1;
                s.staged += n;
                off += n;
            }
        }

        /// A ring offset with `n` free bytes: completed regions are dropped first, then any region the new one overlaps
        /// is waited for (regions on other streams have no order with each other, so each is checked).
        fn reserve(s: *Self, n: usize) G.Error!usize {
            try s.reap(false);
            const size = std.mem.alignForward(usize, n, align_bytes);
            var at = s.head;
            if (at + size > s.ring.bytes.len) at = 0;
            if (s.q_len == max_inflight) try s.retire(0);
            var i: usize = 0;
            while (i < s.q_len) {
                if (s.q_start[i] < at + size and at < s.q_end[i]) {
                    s.waits += 1;
                    try s.retire(i);
                } else i += 1;
            }
            s.head = at + size;
            return at;
        }

        /// Drops finished regions from the front (block: wait for every region).
        fn reap(s: *Self, block: bool) G.Error!void {
            var i: usize = 0;
            while (i < s.q_len) {
                const ev = &s.events[s.q_event[i]];
                if (block) try ev.synchronize() else if (!try ev.done()) {
                    i += 1;
                    continue;
                }
                s.remove(i);
            }
        }

        fn retire(s: *Self, i: usize) G.Error!void {
            try s.events[s.q_event[i]].synchronize();
            s.remove(i);
        }

        fn remove(s: *Self, i: usize) void {
            s.free_events[s.free_len] = s.q_event[i];
            s.free_len += 1;
            for (i + 1..s.q_len) |j| {
                s.q_start[j - 1] = s.q_start[j];
                s.q_end[j - 1] = s.q_end[j];
                s.q_event[j - 1] = s.q_event[j];
            }
            s.q_len -= 1;
        }

        /// Waits for every staged copy (tests, shutdown).
        pub fn drain(s: *Self) G.Error!void {
            try s.reap(true);
        }
    };
}
