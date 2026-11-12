//! sf-disk's namespaces on the host: what a sweep may count and remove under the root, and the IDENTITY lock.
const std = @import("std");
const pd = @import("prompt_disk.zig");
const Key = pd.Key;
const Tier = pd.Tier;
const c = std.c;

const planes = [_]u64{ 96, 32 };
const slot_bytes = 64;
const B = 16; // token bytes a block
const decoy_bytes = 64 << 10; // a file no sweep may count or remove

fn path(buf: []u8, comptime fmt: []const u8, args: anytype) [:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, fmt, args, 0) catch unreachable;
}

fn rmTree(root: []const u8) void {
    var buf: [512]u8 = undefined;
    const d = c.opendir(path(&buf, "{s}", .{root})) orelse return;
    while (c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        var sub: [512]u8 = undefined;
        const p = path(&sub, "{s}/{s}", .{ root, name });
        if (c.unlink(p) != 0) rmTree(p); // a link goes as a link
    }
    _ = c.closedir(d);
    _ = c.rmdir(path(&buf, "{s}", .{root}));
}

fn exists(p: [:0]const u8) bool {
    const fd = c.open(p, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
    if (fd < 0) return false;
    _ = c.close(fd);
    return true;
}

fn mkdir(p: [:0]const u8) !void {
    if (c.mkdir(p, 0o700) != 0) return error.TestDir;
}

fn writeFile(p: [:0]const u8, bytes: []const u8) !void {
    const fd = c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(c.mode_t, 0o600));
    if (fd < 0) return error.TestWrite;
    defer _ = c.close(fd);
    if (c.write(fd, bytes.ptr, bytes.len) != bytes.len) return error.TestWrite;
}

fn nsHex(identity: []const u8) [32]u8 {
    return std.fmt.bytesToHex(pd.namespaceOf(identity), .lower);
}

/// A tier of `identity` under `root` whose budget holds one state of six blocks.
fn openSmall(root: []const u8, identity: []const u8) !*Tier {
    const blk = pd.head_bytes + pd.sum(&planes);
    const budget = 6 * blk + pd.head_bytes + 6 * 32 + slot_bytes + 100;
    return Tier.open(std.testing.allocator, .{ .root = root, .budget = budget, .floor = 0, .queue = 1 << 20, .rate = 0, .free_override = 1 << 40 }, identity, 0, slot_bytes, &planes);
}

/// A six-block state of `identity` saved and written; true when its plan stood.
fn saveSix(t: *Tier, identity: []const u8) !bool {
    const gpa = std.testing.allocator;
    var toks: [24]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast(i);
    var keys: [6]Key = undefined;
    pd.chain(pd.seed(identity), std.mem.sliceAsBytes(&toks), B, &keys);
    var p = try t.plan(&keys, true, false);
    defer p.deinit(gpa);
    if (p.result != .ok) return false;
    for (p.runs) |r| {
        const data = try gpa.alloc(u8, r.n * pd.sum(&planes));
        @memset(data, 7);
        try t.submitRun(keys[r.first..][0..r.n], r.first + 1, data);
    }
    const slot = try gpa.alloc(u8, slot_bytes);
    @memset(slot, 9);
    try t.submitState(&keys, slot, 6);
    try std.testing.expect(t.flush(5 * std.time.ns_per_s));
    return true;
}

test "a sweep never follows a link and removes only a namespace with its own sf-disk IDENTITY" {
    var rb: [128]u8 = undefined;
    const root = path(&rb, "/tmp/tf-sfdisk-ns-{d}", .{c.getpid()});
    var ob: [128]u8 = undefined;
    const out = path(&ob, "/tmp/tf-sfdisk-ns-out-{d}", .{c.getpid()});
    rmTree(root);
    rmTree(out);
    defer rmTree(root);
    defer rmTree(out);
    var pb: [512]u8 = undefined;
    const decoy = try std.testing.allocator.alloc(u8, decoy_bytes);
    defer std.testing.allocator.free(decoy);
    @memset(decoy, 1);
    // a directory outside the root, with a valid IDENTITY for the link that names it under the root
    try mkdir(out);
    try writeFile(path(&pb, "{s}/data", .{out}), decoy);
    try writeFile(path(&pb, "{s}/IDENTITY", .{out}), "linked");
    // a stale namespace (its tier closed) with a link to that directory among its blocks
    var stale: [32]u8 = undefined;
    {
        const t = try openSmall(root, "stale");
        defer t.close();
        stale = t.ns_hex;
        try std.testing.expect(try saveSix(t, "stale"));
    }
    var lb: [512]u8 = undefined;
    if (c.symlink(out, path(&lb, "{s}/{s}/b/zz", .{ root, &stale })) != 0) return error.TestLink;
    if (c.symlink(out, path(&lb, "{s}/{s}", .{ root, &nsHex("linked") })) != 0) return error.TestLink;
    // a 32-hex directory without an IDENTITY, and one whose IDENTITY is not its own sf-disk identity
    try mkdir(path(&pb, "{s}/0123456789abcdef0123456789abcdef", .{root}));
    try writeFile(path(&pb, "{s}/0123456789abcdef0123456789abcdef/data", .{root}), decoy);
    try mkdir(path(&pb, "{s}/fedcba9876543210fedcba9876543210", .{root}));
    try writeFile(path(&pb, "{s}/fedcba9876543210fedcba9876543210/IDENTITY", .{root}), "format=1\nrank=0\n");
    try writeFile(path(&pb, "{s}/fedcba9876543210fedcba9876543210/data", .{root}), decoy);

    const t = try openSmall(root, "new");
    defer t.close();
    try std.testing.expect(t.others_bytes > 0 and t.others_bytes < decoy_bytes); // the stale namespace alone
    try std.testing.expect(try saveSix(t, "new"));
    try std.testing.expect(!exists(path(&pb, "{s}/{s}/IDENTITY", .{ root, &stale })));
    try std.testing.expect(!exists(path(&pb, "{s}/{s}", .{ root, &stale })));
    try std.testing.expect(exists(path(&pb, "{s}/data", .{out})));
    try std.testing.expect(exists(path(&pb, "{s}/IDENTITY", .{out})));
    try std.testing.expect(exists(path(&pb, "{s}/{s}/data", .{ root, &nsHex("linked") })));
    try std.testing.expect(exists(path(&pb, "{s}/0123456789abcdef0123456789abcdef/data", .{root})));
    try std.testing.expect(exists(path(&pb, "{s}/fedcba9876543210fedcba9876543210/data", .{root})));
}

test "a namespace in use is never swept: a tier open on it, or its IDENTITY locked when the sweep runs" {
    var rb: [128]u8 = undefined;
    const root = path(&rb, "/tmp/tf-sfdisk-lock-{d}", .{c.getpid()});
    rmTree(root);
    defer rmTree(root);
    var pb: [512]u8 = undefined;
    const busy = try openSmall(root, "busy");
    defer busy.close();
    try std.testing.expect(try saveSix(busy, "busy"));
    var stale_bytes: u64 = 0;
    var stale: [32]u8 = undefined;
    {
        const t = try openSmall(root, "stale");
        defer t.close();
        stale = t.ns_hex;
        try std.testing.expect(try saveSix(t, "stale"));
        stale_bytes = t.bytes + "stale".len;
    }
    // a second tier on a namespace in use shares it; one whose IDENTITY is held exclusive does not open
    const again = try openSmall(root, "busy");
    again.close();
    const sfd = c.open(path(&pb, "{s}/{s}/IDENTITY", .{ root, &stale }), .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
    try std.testing.expect(sfd >= 0);
    defer _ = c.close(sfd);
    try std.testing.expectEqual(@as(c_int, 0), c.flock(sfd, c.LOCK.EX | c.LOCK.NB));
    try std.testing.expectError(error.NamespaceBusy, openSmall(root, "stale"));
    try std.testing.expectEqual(@as(c_int, 0), c.flock(sfd, c.LOCK.UN));

    const t = try openSmall(root, "new");
    defer t.close();
    try std.testing.expectEqual(stale_bytes, t.others_bytes); // the open one is not counted
    // a tier opens the stale namespace after the scan: the sweep the save asks for leaves it
    try std.testing.expectEqual(@as(c_int, 0), c.flock(sfd, c.LOCK.SH | c.LOCK.NB));
    try std.testing.expect(try saveSix(t, "new"));
    try std.testing.expect(exists(path(&pb, "{s}/{s}/IDENTITY", .{ root, &stale })));
    try std.testing.expect(exists(path(&pb, "{s}/{s}/IDENTITY", .{ root, &busy.ns_hex })));
    try std.testing.expectEqual(@as(?u32, 6), blk: {
        var toks: [24]u32 = undefined;
        for (&toks, 0..) |*x, i| x.* = @intCast(i);
        var keys: [6]Key = undefined;
        pd.chain(pd.seed("busy"), std.mem.sliceAsBytes(&toks), B, &keys);
        break :blk busy.longest(&keys, 0, 6);
    });
}
