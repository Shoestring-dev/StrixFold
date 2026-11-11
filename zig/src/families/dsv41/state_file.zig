//! A kept prompt state on one rank's own disk (--learn): head, slot part, rows; refuses another rank or layout.
const std = @import("std");

pub const magic: u64 = 0x3176_7453_3134_7364; // "ds41Stv1"
pub const version: u64 = 1;
const head_words = 8;
pub const head_bytes = head_words * 8;

/// What a state file says it holds.
pub const Head = struct {
    rank: u64,
    at: u64, // positions the state covers
    layout: u64, // Model.rowsLayout: the planes' shape and the slot part's size
    slot: u64, // the slot part's bytes
    rows: u64, // the compressed rows' bytes
};

/// `<dir>/<key>.r<rank>`: the file of state `key` on `rank`'s box.
pub fn path(buf: []u8, dir: []const u8, key: u64, rank: u64) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.r{d}", .{ dir, key, rank }, 0);
}

/// The head of a file whose payload (slot part, then rows) is `payload`.
pub fn encode(h: Head, payload: []const u8) [head_bytes]u8 {
    const words = [head_words]u64{ magic, version, h.rank, h.at, h.layout, h.slot, h.rows, std.hash.Wyhash.hash(0x51a7e, payload) };
    var out: [head_bytes]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u64, out[i * 8 ..][0..8], w, .little);
    return out;
}

/// The payload of a file's bytes, when its head says exactly `want` and the payload is whole; else why not.
pub fn decode(bytes: []const u8, want: Head) ![]const u8 {
    if (bytes.len < head_bytes) return error.StateFileShort;
    var w: [head_words]u64 = undefined;
    for (&w, 0..) |*x, i| x.* = std.mem.readInt(u64, bytes[i * 8 ..][0..8], .little);
    if (w[0] != magic or w[1] != version) return error.StateFileVersion;
    if (w[2] != want.rank or w[3] != want.at or w[4] != want.layout or w[5] != want.slot or w[6] != want.rows) return error.StateFileMismatch;
    const payload = bytes[head_bytes..];
    if (payload.len != want.slot + want.rows) return error.StateFileShort;
    if (std.hash.Wyhash.hash(0x51a7e, payload) != w[7]) return error.StateFileCorrupt;
    return payload;
}

/// `dir` and its parents (0700); a dir that is there already is fine.
pub fn makePath(dir: []const u8) !void {
    var buf: [1100]u8 = undefined;
    const tmp = try std.fmt.bufPrintSentinel(&buf, "{s}", .{dir}, 0);
    for (tmp[1..], 1..) |ch, i| if (ch == '/') {
        buf[i] = 0;
        _ = std.c.mkdir(@ptrCast(&buf), 0o700);
        buf[i] = '/';
    };
    const rc = std.c.mkdir(tmp.ptr, 0o700);
    if (rc != 0 and std.c.errno(rc) != .EXIST) return error.StateDir;
}

/// `head` then `payload` to `p`: written beside it, fsynced, renamed over it (a reader never sees half a file).
pub fn write(p: [:0]const u8, head: []const u8, payload: []const u8) !void {
    var buf: [1104]u8 = undefined;
    const part = try std.fmt.bufPrintSentinel(&buf, "{s}.part", .{p}, 0);
    const fd = std.c.open(part, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.StateWrite;
    {
        errdefer _ = std.c.unlink(part);
        defer _ = std.c.close(fd);
        try writeAll(fd, head);
        try writeAll(fd, payload);
        if (std.c.fsync(fd) != 0) return error.StateWrite;
    }
    if (std.c.rename(part, p) != 0) {
        _ = std.c.unlink(part);
        return error.StateWrite;
    }
}

/// The whole file at `p` (the caller frees it).
pub fn read(gpa: std.mem.Allocator, p: [:0]const u8) ![]u8 {
    const fd = std.c.open(p, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.StateFileMissing;
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0) return error.StateRead;
    const out = try gpa.alloc(u8, @intCast(end));
    errdefer gpa.free(out);
    var done: usize = 0;
    while (done < out.len) {
        const n = std.c.pread(fd, out.ptr + done, out.len - done, @intCast(done));
        if (n <= 0) return error.StateRead;
        done += @intCast(n);
    }
    return out;
}

pub fn remove(p: [:0]const u8) void {
    _ = std.c.unlink(p);
}

fn writeAll(fd: c_int, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + done, bytes.len - done);
        if (n <= 0) return error.StateWrite;
        done += @intCast(n);
    }
}

test "a state file reads back only into what wrote it" {
    const gpa = std.testing.allocator;
    const payload = "slotpart-rows-rows";
    const h: Head = .{ .rank = 1, .at = 4096, .layout = 0xabc, .slot = 8, .rows = 10 };
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    try bytes.appendSlice(gpa, &encode(h, payload));
    try bytes.appendSlice(gpa, payload);
    try std.testing.expectEqualStrings(payload, try decode(bytes.items, h));
    var other = h;
    other.rank = 0; // rank 0 never reads rank 1's half
    try std.testing.expectError(error.StateFileMismatch, decode(bytes.items, other));
    other = h;
    other.at = 2048;
    try std.testing.expectError(error.StateFileMismatch, decode(bytes.items, other));
    other = h;
    other.layout = 0xabd; // another plane layout
    try std.testing.expectError(error.StateFileMismatch, decode(bytes.items, other));
    bytes.items[head_bytes + 3] ^= 1; // a flipped bit
    try std.testing.expectError(error.StateFileCorrupt, decode(bytes.items, h));
    try std.testing.expectError(error.StateFileShort, decode(bytes.items[0 .. bytes.items.len - 1], h));
    try std.testing.expectError(error.StateFileShort, decode(bytes.items[0..10], h));
}

test "a state file goes to disk whole and comes back" {
    const gpa = std.testing.allocator;
    var dbuf: [256]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dbuf, "/tmp/dsv41-state-test-{d}/learned/abc", .{std.c.getpid()});
    try makePath(dir);
    try makePath(dir); // there already: fine
    var pbuf: [1100]u8 = undefined;
    const p = try path(&pbuf, dir, 0x1234, 1);
    try std.testing.expect(std.mem.endsWith(u8, p, "/0000000000001234.r1"));
    const h: Head = .{ .rank = 1, .at = 2048, .layout = 7, .slot = 3, .rows = 4 };
    try write(p, &encode(h, "abcdefg"), "abcdefg");
    const back = try read(gpa, p);
    defer gpa.free(back);
    try std.testing.expectEqualStrings("abcdefg", try decode(back, h));
    remove(p);
    try std.testing.expectError(error.StateFileMissing, read(gpa, p));
}
