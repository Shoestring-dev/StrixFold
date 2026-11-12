//! sf-disk's small file and clock helpers (libc calls on paths and descriptors), shared by the store's parts.
const std = @import("std");
const Allocator = std.mem.Allocator;
const c = std.c;
const linux = std.os.linux;

/// A directory's entries, "." and ".." skipped.
pub const Entries = struct {
    d: *c.DIR,

    pub fn open(path: []const u8) ?Entries {
        var buf: [1100]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}", .{path}, 0) catch return null;
        return .{ .d = c.opendir(p) orelse return null };
    }

    pub fn next(e: *Entries) ?[]const u8 {
        while (c.readdir(e.d)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
            return name;
        }
        return null;
    }

    pub fn close(e: *Entries) void {
        _ = c.closedir(e.d);
    }
};

pub const FileStat = struct { size: u64, mtime: i128 };

pub fn statPath(path: [:0]const u8) ?FileStat {
    const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = c.close(fd);
    return statFd(fd);
}

pub fn statFd(fd: c_int) ?FileStat {
    var sx: linux.Statx = undefined;
    const at_empty_path: u32 = 0x1000;
    if (linux.errno(linux.statx(fd, "", at_empty_path, .{ .SIZE = true, .MTIME = true }, &sx)) != .SUCCESS) return null;
    return .{ .size = sx.size, .mtime = @as(i128, sx.mtime.sec) * std.time.ns_per_s + sx.mtime.nsec };
}

pub fn touchPath(path: [:0]const u8) void {
    const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
    if (fd < 0) return;
    defer _ = c.close(fd);
    _ = c.futimens(fd, null);
}

pub fn syncDir(path: [:0]const u8) void {
    const fd = c.open(path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true }, @as(c.mode_t, 0));
    if (fd < 0) return;
    defer _ = c.close(fd);
    _ = c.fsync(fd);
}

pub fn writeAll(fd: c_int, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = c.write(fd, bytes.ptr + done, bytes.len - done);
        if (n <= 0) {
            if (n < 0 and c.errno(n) == .INTR) continue;
            return error.DiskWrite;
        }
        done += @intCast(n);
    }
}

pub fn readAt(fd: c_int, dest: []u8, at: u64) bool {
    var done: usize = 0;
    while (done < dest.len) {
        const n = c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) {
            if (n < 0 and c.errno(n) == .INTR) continue;
            return false;
        }
        done += @intCast(n);
    }
    return true;
}

/// `dir` and its parents (mode 0700).
pub fn makePath(gpa: Allocator, dir: [:0]const u8) !void {
    const tmp = try gpa.dupeSentinel(u8, dir, 0);
    defer gpa.free(tmp);
    for (tmp[1..], 1..) |ch, i| if (ch == '/') {
        tmp[i] = 0;
        _ = c.mkdir(tmp.ptr, 0o700);
        tmp[i] = '/';
    };
    const rc = c.mkdir(tmp.ptr, 0o700);
    if (rc != 0 and c.errno(rc) != .EXIST) return error.DiskDir;
}

const Statvfs = extern struct { bsize: c_ulong, frsize: c_ulong, blocks: u64, bfree: u64, bavail: u64, files: u64, ffree: u64, favail: u64, fsid: c_ulong, flag: c_ulong, namemax: c_ulong, spare: [8]c_int };
extern "c" fn statvfs(path: [*:0]const u8, buf: *Statvfs) c_int;

/// The bytes an unprivileged writer may still use on `path`'s filesystem (0 when it cannot tell: no writes).
pub fn fsFree(path: [:0]const u8) u64 {
    var s: Statvfs = undefined;
    if (statvfs(path, &s) != 0) return 0;
    return s.bavail * s.frsize;
}

/// Wall-clock ns (a state file's mtime is the same clock).
pub fn now() i128 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.REALTIME, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

pub fn ms(ns: i128) u64 {
    return @intCast(@max(0, @divTrunc(ns, std.time.ns_per_ms)));
}

pub fn sleepNs(ns: u64) void {
    const ts: c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = c.nanosleep(&ts, null);
}

/// The writer thread in the idle I/O class and at nice 10: a decode round's Engram reads go first.
pub fn lowerPriority() void {
    const idle_class: usize = 3 << 13;
    _ = linux.syscall3(.ioprio_set, 1, 0, idle_class); // IOPRIO_WHO_PROCESS, this thread
    _ = linux.syscall3(.setpriority, 0, @intCast(linux.gettid()), 10); // PRIO_PROCESS on a thread id: this thread
}
