//! sf-disk's file formats: the header every file carries, the files' paths, the writer's FIFO and the reconcile bitmap.
const std = @import("std");
const Allocator = std.mem.Allocator;
const XxHash3 = std.hash.XxHash3;
const pd = @import("prompt_disk.zig");
const Key = pd.Key;
const Tier = pd.Tier;

pub const version: u32 = 1;
/// Every file's header: magic, version, namespace, key, index, rank, payload bytes, payload sum, header sum.
pub const head_bytes = 88;
const magic_block = "SFDB".*;
const magic_state = "SFDS".*;

pub const Kind = enum { block, state };

pub const Head = struct {
    kind: Kind,
    ns: [16]u8,
    key: Key,
    index: u32, // a block's j, a state's k
    rank: u32,
    payload: u64,
    sum: u64,

    pub fn encode(h: Head, out: *[head_bytes]u8) void {
        out[0..4].* = if (h.kind == .block) magic_block else magic_state;
        std.mem.writeInt(u32, out[4..8], version, .little);
        out[8..24].* = h.ns;
        out[24..56].* = h.key;
        std.mem.writeInt(u32, out[56..60], h.index, .little);
        std.mem.writeInt(u32, out[60..64], h.rank, .little);
        std.mem.writeInt(u64, out[64..72], h.payload, .little);
        std.mem.writeInt(u64, out[72..80], h.sum, .little);
        std.mem.writeInt(u64, out[80..88], XxHash3.hash(0, out[0..80]), .little);
    }

    pub fn decode(b: *const [head_bytes]u8) !Head {
        if (std.mem.readInt(u64, b[80..88], .little) != XxHash3.hash(0, b[0..80])) return error.BadHeader;
        const kind: Kind = if (std.mem.eql(u8, b[0..4], &magic_block)) .block else if (std.mem.eql(u8, b[0..4], &magic_state)) .state else return error.BadHeader;
        if (std.mem.readInt(u32, b[4..8], .little) != version) return error.BadHeader;
        return .{ .kind = kind, .ns = b[8..24].*, .key = b[24..56].*, .index = std.mem.readInt(u32, b[56..60], .little), .rank = std.mem.readInt(u32, b[60..64], .little), .payload = std.mem.readInt(u64, b[64..72], .little), .sum = std.mem.readInt(u64, b[72..80], .little) };
    }
};

/// A FIFO whose taken prefix is dropped once it is at least as long as the rest (and 32): storage follows the backlog.
pub fn Fifo(comptime T: type) type {
    return struct {
        items: std.ArrayList(T) = .empty,
        head: usize = 0,

        const Self = @This();

        pub fn len(f: *const Self) usize {
            return f.items.items.len - f.head;
        }

        pub fn push(f: *Self, gpa: Allocator, x: T) !void {
            f.compact();
            try f.items.append(gpa, x);
        }

        pub fn pop(f: *Self) ?T {
            if (f.head == f.items.items.len) return null;
            const x = f.items.items[f.head];
            f.head += 1;
            if (f.head == f.items.items.len) {
                f.items.clearRetainingCapacity();
                f.head = 0;
            }
            return x;
        }

        fn compact(f: *Self) void {
            if (f.head < 32 or f.head < f.len()) return;
            const left = f.len();
            std.mem.copyForwards(T, f.items.items[0..left], f.items.items[f.head..]);
            f.items.shrinkRetainingCapacity(left);
            f.head = 0;
        }

        /// The items still queued.
        pub fn rest(f: *const Self) []T {
            return f.items.items[f.head..];
        }

        pub fn deinit(f: *Self, gpa: Allocator) void {
            f.items.deinit(gpa);
        }
    };
}

/// Block j's file under the key H_j: b/xx/<H_j>.blk, xx its first byte in hex.
pub fn blockPath(t: *const Tier, buf: []u8, k: Key, tmp: bool) ![:0]u8 {
    const hex = std.fmt.bytesToHex(k, .lower);
    return std.fmt.bufPrintSentinel(buf, "{s}/b/{s}/{s}.blk{s}", .{ t.dir, hex[0..2], &hex, if (tmp) ".tmp" else "" }, 0);
}

/// The state file at block j: s/<H_j>.st, listing H_1 .. H_j before the state's own bytes.
pub fn statePath(t: *const Tier, buf: []u8, k: Key, tmp: bool) ![:0]u8 {
    const hex = std.fmt.bytesToHex(k, .lower);
    return std.fmt.bufPrintSentinel(buf, "{s}/s/{s}.st{s}", .{ t.dir, &hex, if (tmp) ".tmp" else "" }, 0);
}

/// The keys of the reconcile's other side, as a bitmap of `have` over `keys` (bit i: keys[i] is on disk here).
pub fn bitmap(gpa: Allocator, t: *const Tier, keys: []const Key) ![]u8 {
    const out = try gpa.alloc(u8, (keys.len + 7) / 8);
    @memset(out, 0);
    for (keys, 0..) |k, i| if (t.hasComplete(k)) {
        out[i / 8] |= @as(u8, 1) << @intCast(i % 8);
    };
    return out;
}

pub fn bit(map: []const u8, i: usize) bool {
    return i / 8 < map.len and map[i / 8] & (@as(u8, 1) << @intCast(i % 8)) != 0;
}

/// The key a file's name holds (64 hex digits then `ext`), or null.
pub fn keyName(name: []const u8, ext: []const u8) ?Key {
    if (name.len != 64 + ext.len or !std.mem.endsWith(u8, name, ext)) return null;
    var k: Key = undefined;
    _ = std.fmt.hexToBytes(&k, name[0..64]) catch return null;
    return k;
}
