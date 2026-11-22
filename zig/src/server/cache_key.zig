//! --cache-transfer-key FILE: the key an exported prompt state is signed with and the bearer the /cache routes take.
const std = @import("std");
const posix = std.posix;
const auth = @import("auth.zig");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The file: the HMAC key as 64 hex digits on one line, the bearer on the next; `#` lines are comments.
pub const Key = struct {
    mac: api.cache_wire.Key,
    bearer: [32]u8, // the bearer's SHA-256 (the text itself is never kept)

    /// The file read once at startup; it must be a regular file no other user can read (chmod 600), under 64 KiB.
    pub fn load(gpa: Allocator, path: []const u8, problem: *[]const u8) !Key {
        const pathz = try gpa.dupeSentinel(u8, path, 0);
        defer gpa.free(pathz);
        const fd = posix.openatZ(posix.AT.FDCWD, pathz, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .NONBLOCK = true }, 0) catch return error.Unsafe;
        defer _ = posix.system.close(fd);
        const st = auth.meta(fd, null) orelse return error.Unsafe;
        if ((st.mode & posix.S.IFMT) != posix.S.IFREG or st.mode & 0o044 != 0) {
            problem.* = "the cache transfer key file must be a regular file unreadable by other users; use chmod 600";
            return error.Invalid;
        }
        if (st.size > 1 << 16) {
            problem.* = "the cache transfer key file exceeds 64 KiB";
            return error.Invalid;
        }
        var text: [1 << 16]u8 = undefined;
        var got: usize = 0;
        while (got < text.len) {
            const n = posix.read(fd, text[got..]) catch return error.Unsafe;
            if (n == 0) break;
            got += n;
        }
        return parse(text[0..got], problem);
    }

    /// The two values in order, blank and `#` lines skipped.
    pub fn parse(text: []const u8, problem: *[]const u8) error{Invalid}!Key {
        var k: Key = undefined;
        var seen: usize = 0;
        var lines = std.mem.splitAny(u8, text, "\n\r");
        while (lines.next()) |raw| {
            const line = std.mem.trim(u8, raw, " \t");
            if (line.len == 0 or line[0] == '#') continue;
            if (seen == 0) {
                if (line.len != 64) return fail(problem, "the cache transfer key file's first line is the HMAC key: 64 hex digits");
                _ = std.fmt.hexToBytes(&k.mac, line) catch return fail(problem, "the cache transfer key file's first line is the HMAC key: 64 hex digits");
            } else if (seen == 1) {
                for (line) |ch| if (ch <= ' ' or ch == 0x7f) return fail(problem, "the cache transfer key file's second line is the bearer: text without whitespace");
                Sha256.hash(line, &k.bearer, .{});
            } else return fail(problem, "the cache transfer key file has more than two values");
            seen += 1;
        }
        if (seen < 2) return fail(problem, "the cache transfer key file needs two lines: the HMAC key (64 hex digits), then the bearer");
        return k;
    }

    fn fail(problem: *[]const u8, message: []const u8) error{Invalid} {
        problem.* = message;
        return error.Invalid;
    }

    /// Whether a request carries the bearer: `Authorization: Bearer` or `x-cache-transfer-key`, compared by digest.
    pub fn accepts(k: *const Key, authorization: ?[]const u8, header: ?[]const u8) bool {
        var ok = false;
        if (authorization) |value| {
            if (std.mem.indexOfScalar(u8, value, ' ')) |space| {
                if (std.ascii.eqlIgnoreCase(value[0..space], "bearer")) ok = k.matches(std.mem.trim(u8, value[space + 1 ..], " \t")) or ok;
            }
        }
        if (header) |value| ok = k.matches(std.mem.trim(u8, value, " \t")) or ok;
        return ok;
    }

    fn matches(k: *const Key, text: []const u8) bool {
        var d: [32]u8 = undefined;
        Sha256.hash(text, &d, .{});
        return std.crypto.timing_safe.eql([32]u8, k.bearer, d);
    }
};

test "the key file: two values, the bearer matched by digest in either header; anything else refused with a reason" {
    var problem: []const u8 = "";
    const hex = "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b";
    const k = try Key.parse("# the pair's key\n" ++ hex ++ "\n  sk-move-1  \n", &problem);
    try std.testing.expectEqual(@as(u8, 0x0b), k.mac[31]);
    try std.testing.expect(k.accepts("Bearer sk-move-1", null));
    try std.testing.expect(k.accepts("bearer  sk-move-1 ", null));
    try std.testing.expect(k.accepts(null, "sk-move-1"));
    try std.testing.expect(k.accepts("Bearer sk-api", "sk-move-1")); // the API key in one header, the bearer in the other
    try std.testing.expect(!k.accepts("Bearer sk-move-2", null));
    try std.testing.expect(!k.accepts("Basic sk-move-1", null));
    try std.testing.expect(!k.accepts(null, null));
    for ([_][]const u8{ "", hex, hex ++ "\nbad bearer\n", "zz" ++ hex[2..] ++ "\nsk\n", hex ++ "\nsk\nthird\n", "short\nsk\n" }) |bad| {
        try std.testing.expectError(error.Invalid, Key.parse(bad, &problem));
        try std.testing.expect(problem.len > 0);
    }
}

test "the key file is refused while others can read it, and read at 0600" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "move", .data = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a\nsk-move\n" });
    const path = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/move", .{tmp.sub_path}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, path, 0o644, 0));
    var problem: []const u8 = "";
    try std.testing.expectError(error.Invalid, Key.load(a, path, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "chmod 600") != null);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, path, 0o600, 0));
    const k = try Key.load(a, path, &problem);
    try std.testing.expect(k.accepts("Bearer sk-move", null));
    try std.testing.expectError(error.Unsafe, Key.load(a, ".zig-cache/tmp/none-such", &problem));
}
