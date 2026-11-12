//! sf-disk's writer thread: the queued runs, states, deletes and sweeps, written under a rate cap and synced.
const std = @import("std");
const Allocator = std.mem.Allocator;
const c = std.c;
const linux = std.os.linux;
const XxHash3 = std.hash.XxHash3;
const pd = @import("prompt_disk.zig");
const format = @import("prompt_disk_format.zig");
const fs = @import("prompt_disk_fs.zig");
const namespace = @import("prompt_disk_namespace.zig");
const Key = pd.Key;
const Tier = pd.Tier;
const Job = pd.Job;
const Reason = pd.Reason;
const head_bytes = format.head_bytes;
const now = fs.now;
const short = pd.short;

/// A write call at most (the rate cap sleeps between them).
const piece_bytes = 8 << 20;

/// A job queued for the writer; one that finds no room fails at once (its bytes freed, its reservation released).
pub fn push(t: *Tier, j: Job) void {
    _ = c.pthread_mutex_lock(&t.mu);
    defer _ = c.pthread_mutex_unlock(&t.mu);
    t.jobs.push(t.gpa, j) catch {
        var x = j;
        t.queued -|= charge(&x);
        t.done.append(t.gpa, .{ .job = x, .ok = false }) catch freeJob(t.gpa, &x);
        return;
    };
    _ = c.pthread_cond_signal(&t.cv);
}

pub fn pushDelete(t: *Tier, states: []Key, blocks: []Key) void {
    if (states.len == 0 and blocks.len == 0) return;
    push(t, .{ .delete = .{ .states = states, .blocks = blocks } });
}

/// The writer thread: jobs in order until stopped and empty (or past the close's deadline).
pub fn writer(t: *Tier) void {
    fs.lowerPriority();
    while (true) {
        _ = c.pthread_mutex_lock(&t.mu);
        while (t.jobs.len() == 0 and !t.stop) _ = c.pthread_cond_wait(&t.cv, &t.mu);
        if (t.jobs.len() == 0 or (t.stop and now() > t.deadline)) {
            _ = c.pthread_mutex_unlock(&t.mu);
            return;
        }
        var j = t.jobs.pop().?;
        t.running = true;
        _ = c.pthread_mutex_unlock(&t.mu);
        var reason: Reason = .io;
        const ok = run(t, &j, &reason);
        // the bytes held while queued go now; the job's record stays for drain
        const ch = charge(&j);
        freePayload(t.gpa, &j);
        _ = c.pthread_mutex_lock(&t.mu);
        t.queued -|= ch;
        t.running = false;
        t.done.append(t.gpa, .{ .job = j, .ok = ok, .reason = reason }) catch freeJob(t.gpa, &j);
        _ = c.pthread_mutex_unlock(&t.mu);
    }
}

fn run(t: *Tier, j: *Job, reason: *Reason) bool {
    switch (j.*) {
        .run => |r| {
            const n = r.keys.len;
            var bytes: u64 = 0;
            for (r.keys, 0..) |k, i| {
                if (t.freeBytes() < t.cfg.floor + head_bytes + t.block_bytes) {
                    reason.* = .floor;
                    t.line("event=blocks result=failed reason=floor count={d}", .{n});
                    return false;
                }
                var parts: [64][]const u8 = undefined;
                if (t.planes.len > parts.len) return false;
                for (t.planes, 0..) |pbytes, p| parts[p] = r.data[pd.rangeOffset(t.planes, n, p) + i * pbytes ..][0..pbytes];
                writeBlock(t, k, r.first_j + @as(u32, @intCast(i)), parts[0..t.planes.len]) catch |e| {
                    t.line("event=blocks result=failed reason={s} count={d}", .{ @errorName(e), n });
                    return false;
                };
                bytes += head_bytes + t.block_bytes;
            }
            syncBlockDirs(t, r.keys);
            _ = t.written.fetchAdd(bytes, .monotonic);
            t.line("event=blocks result=written tokens={d} count={d} bytes={d} ms={d}", .{ (r.first_j - 1) * t.cfg.block_tokens, n, bytes, fs.ms(now() - r.t0) });
            return true;
        },
        .state => |s| {
            // its blocks were written before it (the queue's order): each must be here, whole
            for (s.keys) |k| {
                var pb: [1400]u8 = undefined;
                const st = fs.statPath(t.blockPath(&pb, k, false) catch return false) orelse {
                    reason.* = .io;
                    t.line("event=save result=failed reason=block-missing tokens={d} key={s}", .{ s.k * t.cfg.block_tokens, &short(s.key) });
                    return false;
                };
                if (st.size != head_bytes + t.block_bytes) return false;
            }
            const file = t.stateFile(s.k);
            if (t.freeBytes() < t.cfg.floor + file) {
                reason.* = .floor;
                t.line("event=save result=failed reason=floor tokens={d} key={s}", .{ s.k * t.cfg.block_tokens, &short(s.key) });
                return false;
            }
            writeState(t, s.key, s.k, s.keys, s.slot) catch |e| {
                t.line("event=save result=failed reason={s} tokens={d} key={s}", .{ @errorName(e), s.k * t.cfg.block_tokens, &short(s.key) });
                return false;
            };
            _ = t.written.fetchAdd(file, .monotonic);
            t.line("event=save result=written tokens={d} blocks={d} new_blocks={d} bytes={d} ms={d} key={s}", .{ s.k * t.cfg.block_tokens, s.k, s.new_blocks, s.new_bytes, fs.ms(now() - s.t0), &short(s.key) });
            return true;
        },
        .delete => |d| {
            for (d.states) |k| {
                var pb: [1400]u8 = undefined;
                _ = c.unlink(t.statePath(&pb, k, false) catch continue);
            }
            for (d.blocks) |k| {
                var pb: [1400]u8 = undefined;
                _ = c.unlink(t.blockPath(&pb, k, false) catch continue);
            }
            return true;
        },
        .sweep => |s| return namespace.sweep(t, s.name),
    }
}

fn writeBlock(t: *Tier, k: Key, j: u32, parts: []const []const u8) !void {
    var x = XxHash3.init(0);
    for (parts) |p| x.update(p);
    var hb: [head_bytes]u8 = undefined;
    (format.Head{ .kind = .block, .ns = t.ns, .key = k, .index = j, .rank = t.rank, .payload = t.block_bytes, .sum = x.final() }).encode(&hb);
    var tb: [1400]u8 = undefined;
    var fb: [1400]u8 = undefined;
    const tmp = try t.blockPath(&tb, k, true);
    const hex = std.fmt.bytesToHex(k, .lower);
    var db: [1200]u8 = undefined;
    const d = try std.fmt.bufPrintSentinel(&db, "{s}/b/{s}", .{ t.dir, hex[0..2] }, 0);
    const rc = c.mkdir(d, 0o700);
    if (rc != 0 and c.errno(rc) != .EXIST) return error.DiskDir;
    var all: [65][]const u8 = undefined;
    all[0] = &hb;
    @memcpy(all[1..][0..parts.len], parts);
    try writeThrottled(t, tmp, all[0 .. parts.len + 1]);
    if (c.rename(tmp, try t.blockPath(&fb, k, false)) != 0) {
        _ = c.unlink(tmp);
        return error.DiskWrite;
    }
}

fn writeState(t: *Tier, key: Key, k: u32, keys: []const Key, slot: []const u8) !void {
    var x = XxHash3.init(0);
    x.update(std.mem.sliceAsBytes(keys));
    x.update(slot);
    var hb: [head_bytes]u8 = undefined;
    (format.Head{ .kind = .state, .ns = t.ns, .key = key, .index = k, .rank = t.rank, .payload = t.slot_bytes, .sum = x.final() }).encode(&hb);
    var tb: [1400]u8 = undefined;
    var fb: [1400]u8 = undefined;
    const tmp = try t.statePath(&tb, key, true);
    try writeThrottled(t, tmp, &.{ &hb, std.mem.sliceAsBytes(keys), slot });
    if (c.rename(tmp, try t.statePath(&fb, key, false)) != 0) {
        _ = c.unlink(tmp);
        return error.DiskWrite;
    }
    var db: [1200]u8 = undefined;
    fs.syncDir(std.fmt.bufPrintSentinel(&db, "{s}/s", .{t.dir}, 0) catch return);
}

/// `parts` into a new file at `path`, in pieces under the rate cap, then fdatasync and out of the page cache.
fn writeThrottled(t: *Tier, path: [:0]const u8, parts: []const []const u8) !void {
    const fd = c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(c.mode_t, 0o600));
    if (fd < 0) return error.DiskWrite;
    var ok = false;
    defer {
        _ = c.close(fd);
        if (!ok) _ = c.unlink(path);
    }
    const t0 = now();
    var done: u64 = 0;
    for (parts) |p| {
        var off: usize = 0;
        while (off < p.len) {
            const n = @min(piece_bytes, p.len - off);
            try fs.writeAll(fd, p[off..][0..n]);
            off += n;
            done += n;
            if (t.cfg.rate > 0) {
                const want: i128 = @intCast(done * std.time.ns_per_s / t.cfg.rate);
                const ahead = want - (now() - t0);
                if (ahead > std.time.ns_per_ms) fs.sleepNs(@intCast(ahead));
            }
        }
    }
    if (c.fdatasync(fd) != 0) return error.DiskWrite;
    _ = linux.fadvise(fd, 0, 0, linux.POSIX_FADV.DONTNEED);
    ok = true;
}

fn syncBlockDirs(t: *Tier, keys: []const Key) void {
    var seen: [256]bool = @splat(false);
    for (keys) |k| {
        if (seen[k[0]]) continue;
        seen[k[0]] = true;
        const hex = std.fmt.bytesToHex([1]u8{k[0]}, .lower);
        var db: [1200]u8 = undefined;
        fs.syncDir(std.fmt.bufPrintSentinel(&db, "{s}/b/{s}", .{ t.dir, &hex }, 0) catch continue);
    }
}

pub fn charge(j: *const Job) u64 {
    return switch (j.*) {
        .run => |r| r.charge,
        .state => |s| s.charge,
        else => 0,
    };
}

pub fn freePayload(gpa: Allocator, j: *Job) void {
    switch (j.*) {
        .run => |*r| {
            gpa.free(r.data);
            r.data = &.{};
        },
        .state => |*s| {
            gpa.free(s.slot);
            s.slot = &.{};
        },
        else => {},
    }
}

pub fn freeJob(gpa: Allocator, j: *Job) void {
    freePayload(gpa, j);
    switch (j.*) {
        .run => |r| gpa.free(r.keys),
        .state => |s| gpa.free(s.keys),
        .delete => |d| {
            gpa.free(d.states);
            gpa.free(d.blocks);
        },
        .sweep => {},
    }
}
