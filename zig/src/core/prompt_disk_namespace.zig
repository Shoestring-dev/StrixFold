//! sf-disk's namespaces under the root: this one's IDENTITY claimed and locked, other ones counted and swept for room.
const std = @import("std");
const c = std.c;
const linux = std.os.linux;
const pd = @import("prompt_disk.zig");
const fs = @import("prompt_disk_fs.zig");
const Tier = pd.Tier;

/// The deepest a sweep walks below a namespace (s/, b/xx/); deeper entries stay, and so does its IDENTITY.
const max_depth = 4;

/// IDENTITY, held shared while the tier is open (a sweep takes it exclusive); another identity here is refused.
pub fn claim(t: *Tier, identity: []const u8) !void {
    var buf: [1100]u8 = undefined;
    const p = try std.fmt.bufPrintSentinel(&buf, "{s}/IDENTITY", .{t.dir}, 0);
    const fd = c.open(p, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .NONBLOCK = true, .CLOEXEC = true }, @as(c.mode_t, 0));
    if (fd >= 0) {
        errdefer _ = c.close(fd);
        try lockShared(fd);
        if (readIdentity(t.gpa, fd)) |have| {
            defer t.gpa.free(have);
            if (!std.mem.eql(u8, have, identity)) return error.ForeignNamespace;
            _ = c.futimens(fd, null); // its mtime orders the namespaces for a sweep
            t.lock = fd;
            return;
        }
        _ = c.close(fd);
    }
    // a new IDENTITY is locked before its rename: it is never in place unlocked
    var tb: [1100]u8 = undefined;
    const tmp = try std.fmt.bufPrintSentinel(&tb, "{s}/IDENTITY.tmp", .{t.dir}, 0);
    const nfd = c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(c.mode_t, 0o600));
    if (nfd < 0) return error.DiskWrite;
    errdefer _ = c.close(nfd);
    try lockShared(nfd);
    try fs.writeAll(nfd, identity);
    if (c.fdatasync(nfd) != 0 or c.rename(tmp, p) != 0) return error.DiskWrite;
    t.lock = nfd;
}

fn lockShared(fd: c_int) !void {
    if (c.flock(fd, c.LOCK.SH | c.LOCK.NB) == 0) return;
    return if (c.errno(-1) == .AGAIN) error.NamespaceBusy else error.DiskLock;
}

/// The lock on IDENTITY released (close: the tier is gone).
pub fn release(t: *Tier) void {
    if (t.lock >= 0) _ = c.close(t.lock);
    t.lock = -1;
}

/// Other namespaces under the root that a sweep may take: their bytes count against the budget; each logged.
pub fn scanOthers(t: *Tier, identity: []const u8) void {
    const root = openRoot(t) orelse return;
    defer _ = c.close(root);
    const d = openDirStream(root) orelse return;
    defer _ = c.closedir(d);
    while (c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (name.len != 32 or std.mem.eql(u8, name, &t.ns_hex) or !allHex(name)) continue;
        var h = Held.take(t.gpa, root, name[0..32].*) catch |why| {
            t.line("event=foreign ns={s} kept={s}", .{ name, @errorName(why) });
            continue;
        };
        defer h.release(t.gpa);
        const bytes = walk(h.dir, false, max_depth, null).bytes;
        const used: i128 = if (fs.statFd(h.id)) |st| st.mtime else 0;
        t.others.append(t.gpa, .{ .name = name[0..32].*, .bytes = bytes, .used = used }) catch return;
        t.others_bytes += bytes;
        var diff_buf: [1024]u8 = undefined;
        t.line("event=foreign ns={s} differs={s} mib={d}", .{ name, identityDiff(&diff_buf, identity, h.text), bytes >> 20 });
    }
}

/// Other namespaces removed, least recently opened first, until `files` more fit (or none is left).
pub fn sweepOthers(t: *Tier, files: u64) void {
    while (t.others.items.len > 0 and t.bytes + t.others_bytes + files > t.cfg.budget) {
        var oldest: usize = 0;
        for (t.others.items, 0..) |o, i| if (o.used < t.others.items[oldest].used) {
            oldest = i;
        };
        const o = t.others.swapRemove(oldest);
        t.others_bytes -|= o.bytes;
        t.line("event=sweep ns={s} mib={d}", .{ &o.name, o.bytes >> 20 });
        t.push(.{ .sweep = .{ .name = o.name } });
    }
}

/// The writer's sweep of namespace `name` under its IDENTITY's exclusive lock: the tree, IDENTITY, then the folder.
pub fn sweep(t: *Tier, name: [32]u8) bool {
    const root = openRoot(t) orelse return false;
    defer _ = c.close(root);
    var h = Held.take(t.gpa, root, name) catch |why| {
        t.line("event=sweep result=kept ns={s} reason={s}", .{ &name, @errorName(why) });
        return true;
    };
    defer h.release(t.gpa);
    // states before blocks, IDENTITY last: a sweep cut short leaves a namespace a later one finishes
    const s = walkAt(h.dir, "s", true, max_depth);
    const rest = walk(h.dir, true, max_depth, "IDENTITY");
    if (s.left or rest.left) {
        t.line("event=sweep result=partial ns={s}", .{&name});
        return false;
    }
    if (c.unlinkat(h.dir, "IDENTITY", 0) != 0) return false;
    var nz: [33]u8 = undefined;
    _ = c.unlinkat(root, zName(&nz, name), linux.AT.REMOVEDIR);
    return true;
}

fn zName(buf: *[33]u8, name: [32]u8) [:0]const u8 {
    buf[0..32].* = name;
    buf[32] = 0;
    return buf[0..32 :0];
}

/// A directory stream over a duplicate of `dir` (the descriptor itself stays usable for the *at calls).
fn openDirStream(dir: c_int) ?*c.DIR {
    const fd = c.dup(dir);
    if (fd < 0) return null;
    return c.fdopendir(fd) orelse {
        _ = c.close(fd);
        return null;
    };
}

fn openRoot(t: *Tier) ?c_int {
    var buf: [1100]u8 = undefined;
    const p = std.fmt.bufPrintSentinel(&buf, "{s}", .{t.cfg.root}, 0) catch return null;
    const fd = c.open(p, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, @as(c.mode_t, 0));
    return if (fd < 0) null else fd;
}

/// Another namespace a sweep may count or remove: a directory (not a link) whose IDENTITY names it, held exclusive.
const Held = struct {
    dir: c_int,
    id: c_int,
    text: []u8,

    /// NotADirectory: a link or a file; NoIdentity; NotSfDisk: its IDENTITY's SHA-256 does not name it; InUse: locked.
    fn take(gpa: std.mem.Allocator, root: c_int, name: [32]u8) !Held {
        var nz: [33]u8 = undefined;
        const dir = c.openat(root, zName(&nz, name), .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .NONBLOCK = true, .CLOEXEC = true }, @as(c.mode_t, 0));
        if (dir < 0) return error.NotADirectory;
        errdefer _ = c.close(dir);
        const id = c.openat(dir, "IDENTITY", .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .NONBLOCK = true, .CLOEXEC = true }, @as(c.mode_t, 0));
        if (id < 0) return error.NoIdentity;
        errdefer _ = c.close(id);
        const text = readIdentity(gpa, id) orelse return error.NotSfDisk;
        errdefer gpa.free(text);
        if (!std.mem.eql(u8, &std.fmt.bytesToHex(pd.namespaceOf(text), .lower), &name)) return error.NotSfDisk;
        if (c.flock(id, c.LOCK.EX | c.LOCK.NB) != 0) return error.InUse;
        return .{ .dir = dir, .id = id, .text = text };
    }

    fn release(h: *Held, gpa: std.mem.Allocator) void {
        gpa.free(h.text);
        _ = c.close(h.id);
        _ = c.close(h.dir);
    }
};

/// An IDENTITY's text: a regular file of at most 1 MiB (its size checked before the allocation), or null.
fn readIdentity(gpa: std.mem.Allocator, fd: c_int) ?[]u8 {
    var sx: linux.Statx = undefined;
    const at_empty_path: u32 = 0x1000;
    if (linux.errno(linux.statx(fd, "", at_empty_path, .{ .TYPE = true, .SIZE = true }, &sx)) != .SUCCESS) return null;
    if (!linux.S.ISREG(sx.mode) or sx.size > 1 << 20) return null;
    const out = gpa.alloc(u8, @intCast(sx.size)) catch return null;
    if (!fs.readAt(fd, out, 0)) {
        gpa.free(out);
        return null;
    }
    return out;
}

const Walk = struct { bytes: u64 = 0, left: bool = false };

/// Entry `name` of directory `dir`, never followed: a directory walked (and removed), anything else unlinked as itself.
fn walkAt(dir: c_int, name: [*:0]const u8, remove: bool, depth: u32) Walk {
    var sx: linux.Statx = undefined;
    switch (linux.errno(linux.statx(dir, name, linux.AT.SYMLINK_NOFOLLOW, .{ .TYPE = true, .SIZE = true }, &sx))) {
        .SUCCESS => {},
        .NOENT => return .{},
        else => return .{ .left = true },
    }
    var w: Walk = .{};
    if (linux.S.ISDIR(sx.mode)) {
        if (depth == 0) return .{ .left = true };
        const sub = c.openat(dir, name, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .NONBLOCK = true, .CLOEXEC = true }, @as(c.mode_t, 0));
        if (sub < 0) return .{ .left = true };
        w = walk(sub, remove, depth - 1, null);
        _ = c.close(sub);
        if (remove and !w.left and c.unlinkat(dir, name, linux.AT.REMOVEDIR) != 0) w.left = true;
        return w;
    }
    if (linux.S.ISREG(sx.mode)) w.bytes = sx.size; // a link, a socket or a device counts nothing
    if (remove and c.unlinkat(dir, name, 0) != 0) w.left = true;
    return w;
}

/// The bytes of the regular files below directory `dir` (links never followed), each entry removed when `remove`.
fn walk(dir: c_int, remove: bool, depth: u32, skip: ?[]const u8) Walk {
    const d = openDirStream(dir) orelse return .{ .left = true };
    defer _ = c.closedir(d);
    var w: Walk = .{};
    while (c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        if (skip) |s| if (std.mem.eql(u8, name, s)) continue;
        const e = walkAt(dir, @ptrCast(&ent.name), remove, depth);
        w.bytes += e.bytes;
        w.left = w.left or e.left;
    }
    return w;
}

/// The `key=value` lines of `theirs` that differ from `ours` or are missing from either, comma separated, or "none".
pub fn identityDiff(buf: []u8, ours: []const u8, theirs: []const u8) []const u8 {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, ours, '\n');
    while (it.next()) |l| {
        if (l.len == 0) continue;
        const name = l[0 .. std.mem.indexOfScalar(u8, l, '=') orelse l.len];
        if (lineOf(theirs, name)) |o| if (std.mem.eql(u8, o, l)) continue;
        n = appendName(buf, n, name);
    }
    var jt = std.mem.splitScalar(u8, theirs, '\n');
    while (jt.next()) |l| {
        if (l.len == 0) continue;
        const name = l[0 .. std.mem.indexOfScalar(u8, l, '=') orelse l.len];
        if (lineOf(ours, name) == null) n = appendName(buf, n, name);
    }
    return if (n == 0) "none" else buf[0..n];
}

fn lineOf(text: []const u8, name: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |l| if (l.len > name.len and l[name.len] == '=' and std.mem.startsWith(u8, l, name)) return l else if (std.mem.eql(u8, l, name)) return l;
    return null;
}

fn appendName(buf: []u8, n: usize, name: []const u8) usize {
    const need = name.len + @intFromBool(n > 0);
    if (n + need > buf.len) return n;
    var m = n;
    if (m > 0) {
        buf[m] = ',';
        m += 1;
    }
    @memcpy(buf[m..][0..name.len], name);
    return m + name.len;
}

fn allHex(s: []const u8) bool {
    for (s) |ch| if (!std.ascii.isHex(ch)) return false;
    return true;
}
