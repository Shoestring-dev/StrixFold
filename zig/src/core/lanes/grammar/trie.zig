//! Literal tries: each literal's bytes from a root, its value at the node where it ends. A program keeps all of its
//! tries in two flat arrays (Tries); a Builder makes one and flattens it there.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// No node, no trie.
pub const none: u32 = std.math.maxInt(u32);

/// A flattened node: its edges at `Tries.edges[first..][0..len]` in byte order, and the value of a literal that ends
/// here (-1: none).
pub const Node = struct { first: u32, len: u32, value: i32 };
pub const Edge = struct { byte: u8, next: u32 };

pub const Tries = struct {
    nodes: []const Node = &.{},
    edges: []const Edge = &.{},

    pub fn child(t: *const Tries, node: u32, b: u8) ?u32 {
        const n = t.nodes[node];
        for (t.edges[n.first..][0..n.len]) |e| {
            if (e.byte == b) return e.next;
            if (e.byte > b) break;
        }
        return null;
    }

    /// The value of the literal ending at `node`, or null.
    pub fn value(t: *const Tries, node: u32) ?i32 {
        const v = t.nodes[node].value;
        return if (v == -1) null else v;
    }

    pub fn leaf(t: *const Tries, node: u32) bool {
        return t.nodes[node].len == 0;
    }

    /// Whether a literal starts with `b` from `node`.
    pub fn starts(t: *const Tries, node: u32, b: u8) bool {
        return t.child(node, b) != null;
    }
};

/// The flat arrays a program's tries go into while it is compiled.
pub const Flat = struct {
    nodes: std.ArrayList(Node) = .empty,
    edges: std.ArrayList(Edge) = .empty,

    pub fn tries(f: *const Flat) Tries {
        return .{ .nodes = f.nodes.items, .edges = f.edges.items };
    }
};

/// One trie being built.
pub const Builder = struct {
    kids: std.ArrayList(std.ArrayList(Edge)) = .empty,
    values: std.ArrayList(i32) = .empty,
    count: usize = 0, // literals added

    pub fn init(a: Allocator) !Builder {
        var b: Builder = .{};
        try b.kids.append(a, .empty);
        try b.values.append(a, -1);
        return b;
    }

    /// Adds `bytes` ending in `v` (>= 0, or a caller's own negative mark below -1); a literal added twice keeps its
    /// first value.
    pub fn add(b: *Builder, a: Allocator, bytes: []const u8, v: i32) !void {
        std.debug.assert(v != -1);
        var at: u32 = 0;
        for (bytes) |c| {
            const kids = &b.kids.items[at];
            var next: ?u32 = null;
            for (kids.items) |e| if (e.byte == c) {
                next = e.next;
                break;
            };
            if (next == null) {
                const n: u32 = @intCast(b.kids.items.len);
                try b.kids.append(a, .empty);
                try b.values.append(a, -1);
                try b.kids.items[at].append(a, .{ .byte = c, .next = n });
                next = n;
            }
            at = next.?;
        }
        if (b.values.items[at] == -1) b.values.items[at] = v;
        b.count += 1;
    }

    /// Flattens the trie into `f`; its root's index there.
    pub fn finish(b: *Builder, a: Allocator, f: *Flat) !u32 {
        const base: u32 = @intCast(f.nodes.items.len);
        // node i of the builder becomes base + i: its edges sorted and appended
        for (b.kids.items, b.values.items) |*kids, v| {
            std.mem.sort(Edge, kids.items, {}, struct {
                fn less(_: void, x: Edge, y: Edge) bool {
                    return x.byte < y.byte;
                }
            }.less);
            try f.nodes.append(a, .{ .first = @intCast(f.edges.items.len), .len = @intCast(kids.items.len), .value = v });
            for (kids.items) |e| try f.edges.append(a, .{ .byte = e.byte, .next = base + e.next });
        }
        return base;
    }
};

test "literals end at their values; a prefix of another waits for more" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Flat = .{};
    var b = try Builder.init(a);
    try b.add(a, "yes", 0);
    try b.add(a, "yesterday", 1);
    try b.add(a, "no", 2);
    try b.add(a, "no", 5);
    const root = try b.finish(a, &f);
    const t = f.tries();
    var at = root;
    for ("yes") |c| at = t.child(at, c).?;
    try std.testing.expectEqual(@as(?i32, 0), t.value(at));
    try std.testing.expect(!t.leaf(at));
    at = root;
    for ("no") |c| at = t.child(at, c).?;
    try std.testing.expectEqual(@as(?i32, 2), t.value(at));
    try std.testing.expect(t.leaf(at));
    try std.testing.expectEqual(@as(?u32, null), t.child(root, 'x'));
}
