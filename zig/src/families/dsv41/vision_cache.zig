//! Process-local LRU of exact encoded image rows. Only the lane thread owns the cache.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Key = [Sha256.digest_length]u8;

pub fn key(payload: []const u8) Key {
    var digest: Key = undefined;
    Sha256.hash(payload, &digest, .{});
    return digest;
}

pub const Cache = struct {
    allocator: std.mem.Allocator,
    budget: usize,
    held: usize = 0,
    entries: std.ArrayList(Entry) = .empty,

    const Entry = struct { digest: Key, rows: []u8 };

    pub fn get(c: *Cache, digest: Key) ?[]const u8 {
        for (c.entries.items, 0..) |entry, i| {
            if (!std.mem.eql(u8, &entry.digest, &digest)) continue;
            _ = c.entries.orderedRemove(i);
            c.entries.appendAssumeCapacity(entry);
            return entry.rows;
        }
        return null;
    }

    fn removeOldest(c: *Cache) void {
        const old = c.entries.orderedRemove(0);
        c.held -= old.rows.len;
        c.allocator.free(old.rows);
    }

    pub fn put(c: *Cache, digest: Key, rows: []const u8) void {
        if (rows.len == 0 or rows.len > c.budget or c.get(digest) != null) return;
        while (c.held > c.budget - rows.len) c.removeOldest();
        const copy = c.allocator.dupe(u8, rows) catch return;
        c.entries.append(c.allocator, .{ .digest = digest, .rows = copy }) catch {
            c.allocator.free(copy);
            return;
        };
        c.held += copy.len;
    }

    pub fn deinit(c: *Cache) void {
        for (c.entries.items) |entry| c.allocator.free(entry.rows);
        c.entries.deinit(c.allocator);
    }
};

test "encoded rows match the entire payload and evict the least recently used bytes" {
    var cache: Cache = .{ .allocator = std.testing.allocator, .budget = 8 };
    defer cache.deinit();
    cache.put(key("grid A"), "red!");
    cache.put(key("grid B"), "blue");
    try std.testing.expectEqualStrings("red!", cache.get(key("grid A")).?);
    cache.put(key("grid C"), "gold");
    try std.testing.expect(cache.get(key("grid B")) == null);
    try std.testing.expectEqualStrings("red!", cache.get(key("grid A")).?);
    try std.testing.expectEqual(@as(usize, 8), cache.held);
}
