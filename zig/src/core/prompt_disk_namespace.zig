//! sf-disk's namespaces under the root: this one's IDENTITY claimed, other ones counted and swept for room.
const std = @import("std");
const c = std.c;
const pd = @import("prompt_disk.zig");
const fs = @import("prompt_disk_fs.zig");
const Tier = pd.Tier;
const Entries = fs.Entries;

/// IDENTITY: written when new; another identity in this namespace (a hash prefix collision) is refused.
pub fn claim(t: *Tier, identity: []const u8) !void {
    var buf: [1100]u8 = undefined;
    const p = try std.fmt.bufPrintSentinel(&buf, "{s}/IDENTITY", .{t.dir}, 0);
    if (fs.readSmall(t.gpa, p)) |have| {
        defer t.gpa.free(have);
        if (!std.mem.eql(u8, have, identity)) return error.ForeignNamespace;
        fs.touchPath(p);
        return;
    }
    var tb: [1100]u8 = undefined;
    const tmp = try std.fmt.bufPrintSentinel(&tb, "{s}/IDENTITY.tmp", .{t.dir}, 0);
    try fs.writeFile(tmp, &.{identity});
    if (c.rename(tmp, p) != 0) return error.DiskWrite;
}

/// Other namespaces under the root: their bytes count against the budget; each logged with what differs.
pub fn scanOthers(t: *Tier, identity: []const u8) void {
    var it = Entries.open(t.cfg.root) orelse return;
    defer it.close();
    while (it.next()) |name| {
        if (name.len != 32 or std.mem.eql(u8, name, &t.ns_hex) or !allHex(name)) continue;
        {
            var pb: [1100]u8 = undefined;
            const path = std.fmt.bufPrint(&pb, "{s}/{s}", .{ t.cfg.root, name }) catch continue;
            const bytes = treeBytes(path, false);
            var ib: [1200]u8 = undefined;
            const ip = std.fmt.bufPrintSentinel(&ib, "{s}/IDENTITY", .{path}, 0) catch continue;
            const used: i128 = if (fs.statPath(ip)) |st| st.mtime else 0;
            t.others.append(t.gpa, .{ .name = name[0..32].*, .bytes = bytes, .used = used }) catch return;
            t.others_bytes += bytes;
            var diff_buf: [1024]u8 = undefined;
            const differs = if (fs.readSmall(t.gpa, ip)) |theirs| blk: {
                defer t.gpa.free(theirs);
                break :blk identityDiff(&diff_buf, identity, theirs);
            } else "no-identity";
            t.line("event=foreign ns={s} differs={s} mib={d}", .{ name, differs, bytes >> 20 });
        }
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

/// The writer's sweep of namespace `name`: its files and directories removed.
pub fn sweep(t: *Tier, name: [32]u8) bool {
    var pb: [1100]u8 = undefined;
    const path = std.fmt.bufPrint(&pb, "{s}/{s}", .{ t.cfg.root, &name }) catch return false;
    _ = treeBytes(path, true);
    return true;
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

/// The bytes of the regular files under `path` (s/, b/ and b/xx/), removed with the directories when `remove`.
fn treeBytes(path: []const u8, remove: bool) u64 {
    var total: u64 = 0;
    var it = Entries.open(path) orelse return 0;
    while (it.next()) |name| {
        var pb: [1400]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&pb, "{s}/{s}", .{ path, name }, 0) catch continue;
        if (c.opendir(p)) |d| {
            _ = c.closedir(d);
            total += treeBytes(p, remove); // (removes p itself when `remove`)
        } else if (fs.statPath(p)) |st| {
            total += st.size;
            if (remove) _ = c.unlink(p);
        }
    }
    it.close();
    if (remove) {
        var pb: [1400]u8 = undefined;
        _ = c.rmdir(std.fmt.bufPrintSentinel(&pb, "{s}", .{path}, 0) catch return total);
    }
    return total;
}
