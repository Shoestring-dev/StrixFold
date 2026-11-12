//! sf-disk's index from the files at open, and a resume's verified reads of a state and its blocks.
const std = @import("std");
const c = std.c;
const linux = std.os.linux;
const XxHash3 = std.hash.XxHash3;
const pd = @import("prompt_disk.zig");
const format = @import("prompt_disk_format.zig");
const fs = @import("prompt_disk_fs.zig");
const Key = pd.Key;
const Tier = pd.Tier;
const Head = format.Head;
const head_bytes = format.head_bytes;
const Entries = fs.Entries;
const readAt = fs.readAt;

/// The index rebuilt from the files: temporary files, blocks no state lists and states missing a block are removed.
pub fn scan(t: *Tier) !void {
    // blocks: name and size (headers are read at a resume)
    var bdir: [1100]u8 = undefined;
    for (0..256) |x| {
        const hex = std.fmt.bytesToHex([1]u8{@intCast(x)}, .lower);
        const d = try std.fmt.bufPrint(&bdir, "{s}/b/{s}", .{ t.dir, &hex });
        var it = Entries.open(d) orelse continue;
        defer it.close();
        while (it.next()) |name| {
            var pb: [1400]u8 = undefined;
            const path = std.fmt.bufPrintSentinel(&pb, "{s}/{s}", .{ d, name }, 0) catch continue;
            if (std.mem.endsWith(u8, name, ".tmp")) {
                _ = c.unlink(path);
                continue;
            }
            const key = format.keyName(name, ".blk") orelse continue;
            const st = fs.statPath(path) orelse continue;
            if (st.size != head_bytes + t.block_bytes or !std.mem.eql(u8, name[0..2], &hex)) {
                _ = c.unlink(path);
                continue;
            }
            try t.blocks.put(t.gpa, key, .{ .bytes = st.size, .status = .complete });
        }
    }
    // states: header and block list; a state missing a block, or not this namespace's, goes
    var sd: [1100]u8 = undefined;
    const sdir = try std.fmt.bufPrint(&sd, "{s}/s", .{t.dir});
    var it = Entries.open(sdir) orelse return error.DiskDir;
    defer it.close();
    while (it.next()) |name| {
        var pb: [1400]u8 = undefined;
        const path = std.fmt.bufPrintSentinel(&pb, "{s}/{s}", .{ sdir, name }, 0) catch continue;
        if (std.mem.endsWith(u8, name, ".tmp")) {
            _ = c.unlink(path);
            continue;
        }
        const key = format.keyName(name, ".st") orelse continue;
        const rec = readStateMeta(t, path, key) catch {
            _ = c.unlink(path);
            continue;
        };
        const whole = for (rec.keys) |bk| {
            if (!t.blocks.contains(bk)) break false;
        } else true;
        if (!whole) {
            t.gpa.free(rec.keys);
            _ = c.unlink(path);
            continue;
        }
        for (rec.keys) |bk| t.blocks.getPtr(bk).?.refs += 1;
        t.states.put(t.gpa, key, rec) catch |e| {
            t.gpa.free(rec.keys);
            return e;
        };
        t.clock = @max(t.clock, rec.used);
    }
    // blocks no state lists (a crash between a run and its state, an eviction cut short)
    var gone: std.ArrayList(Key) = .empty;
    defer gone.deinit(t.gpa);
    var bi = t.blocks.iterator();
    while (bi.next()) |e| if (e.value_ptr.refs == 0) try gone.append(t.gpa, e.key_ptr.*);
    for (gone.items) |k| {
        _ = t.blocks.remove(k);
        var pb: [1400]u8 = undefined;
        _ = c.unlink(t.blockPath(&pb, k, false) catch continue);
    }
    var bv = t.blocks.valueIterator();
    while (bv.next()) |b| t.bytes += b.bytes;
    var sv = t.states.valueIterator();
    while (sv.next()) |s| t.bytes += s.bytes;
}

/// A state file's header and block list (its size checked before the list is allocated).
fn readStateMeta(t: *Tier, path: [:0]const u8, key: Key) !pd.StateRec {
    const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
    if (fd < 0) return error.DiskRead;
    defer _ = c.close(fd);
    var hb: [head_bytes]u8 = undefined;
    if (!readAt(fd, &hb, 0)) return error.DiskRead;
    const h = try Head.decode(&hb);
    if (h.kind != .state or !std.mem.eql(u8, &h.ns, &t.ns) or !std.mem.eql(u8, &h.key, &key) or h.rank != t.rank or h.index == 0 or h.index > pd.max_blocks) return error.BadHeader;
    const file = head_bytes + 32 * @as(u64, h.index) + t.slot_bytes;
    const st = fs.statFd(fd) orelse return error.DiskRead;
    if (st.size != file or h.payload != t.slot_bytes) return error.BadSize;
    const keys = try t.gpa.alloc(Key, h.index);
    errdefer t.gpa.free(keys);
    if (!readAt(fd, std.mem.sliceAsBytes(keys), head_bytes)) return error.DiskRead;
    if (!std.mem.eql(u8, &keys[h.index - 1], &key)) return error.BadHeader;
    return .{ .k = h.index, .bytes = file, .used = st.mtime, .status = .complete, .keys = keys };
}

/// State `key` of k blocks read and verified: its keys into `keys`, its own bytes into `slot` (busy: being written).
pub fn readState(t: *Tier, key: Key, k: u32, keys: []Key, slot: []u8) pd.Load {
    t.drain();
    const rec = t.states.get(key) orelse return .bad;
    if (rec.status == .pending) return .busy;
    if (rec.k != k or keys.len != k or slot.len != t.slot_bytes) return .bad;
    var pb: [1400]u8 = undefined;
    const path = t.statePath(&pb, key, false) catch return .bad;
    const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
    if (fd < 0) return .bad;
    defer {
        _ = linux.fadvise(fd, 0, 0, linux.POSIX_FADV.DONTNEED);
        _ = c.close(fd);
    }
    var hb: [head_bytes]u8 = undefined;
    if (!readAt(fd, &hb, 0)) return .bad;
    const h = Head.decode(&hb) catch return .bad;
    if (h.kind != .state or !std.mem.eql(u8, &h.ns, &t.ns) or !std.mem.eql(u8, &h.key, &key) or h.index != k or h.rank != t.rank or h.payload != t.slot_bytes) return .bad;
    if (!readAt(fd, std.mem.sliceAsBytes(keys), head_bytes) or !readAt(fd, slot, head_bytes + 32 * @as(u64, k))) return .bad;
    var x = XxHash3.init(0);
    x.update(std.mem.sliceAsBytes(keys));
    x.update(slot);
    if (x.final() != h.sum or !std.mem.eql(u8, &keys[k - 1], &key)) return .bad;
    return .ok;
}

/// Blocks `keys` (block first_j onward) read and verified into `data` as a run; `scratch` holds one block file.
pub fn readRun(t: *Tier, keys: []const Key, first_j: u32, data: []u8, scratch: []u8) bool {
    return readRunOr(t, keys, first_j, data, scratch, null);
}

/// readRun, naming the block found missing or corrupt in `bad` (when given).
pub fn readRunOr(t: *Tier, keys: []const Key, first_j: u32, data: []u8, scratch: []u8, bad: ?*Key) bool {
    const n = keys.len;
    if (data.len < n * t.block_bytes or scratch.len < head_bytes + t.block_bytes) return false;
    for (keys, 0..) |k, i| {
        if (bad) |b| b.* = k;
        var pb: [1400]u8 = undefined;
        const path = t.blockPath(&pb, k, false) catch return false;
        const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
        if (fd < 0) return false;
        const file = scratch[0 .. head_bytes + t.block_bytes];
        const got = readAt(fd, file, 0);
        var extra: [1]u8 = undefined;
        const longer = readAt(fd, &extra, file.len);
        _ = linux.fadvise(fd, 0, 0, linux.POSIX_FADV.DONTNEED);
        _ = c.close(fd);
        if (!got or longer) return false;
        const h = Head.decode(file[0..head_bytes]) catch return false;
        const payload = file[head_bytes..];
        if (h.kind != .block or !std.mem.eql(u8, &h.ns, &t.ns) or !std.mem.eql(u8, &h.key, &k) or h.index != first_j + i or h.rank != t.rank or h.payload != t.block_bytes) return false;
        if (XxHash3.hash(0, payload) != h.sum) return false;
        var off: u64 = 0;
        for (t.planes, 0..) |pbytes, p| {
            const dst = pd.rangeOffset(t.planes, n, p) + i * pbytes;
            @memcpy(data[dst..][0..pbytes], payload[off..][0..pbytes]);
            off += pbytes;
        }
    }
    return true;
}
