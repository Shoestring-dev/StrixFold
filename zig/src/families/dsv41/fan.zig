//! The ranks' TCP links and a node's place in the split (moved out of prompt2d.zig so a TP2 build without the
//! four-node split has them): rank 0 listens on PORT + g - 1 for follower g, one follower at a time (each follower
//! retries its connect while rank 0 gets to it), so two ranks link exactly as before (rank 1 on PORT). The links carry
//! rank 0's lane commands, the collectives' bootstrap (NCCL's unique id on CUDA) and the peers' handshake.
const std = @import("std");
const plan = @import("plan.zig");
const Link = @import("link.zig").Link;

/// The split of node `g` of `world` nodes: TP over 1 or 2, the 2D split over 4.
pub fn split(g: u32, world: u32) !plan.Split {
    return switch (world) {
        1, 2 => .{ .rank = g, .world = world },
        4 => .{ .rank = g % 2, .world = 2, .pair = g / 2 },
        else => error.UnsupportedWorld,
    };
}

/// How long each side waits for the other to link: rank 1 retries its connect while rank 0 loads, rank 0 waits for
/// each follower's connect after its own load (the ranks load at the same time: a follower that has not connected
/// within this has died or hung, and rank 0's open fails rather than waiting forever, U47).
pub const link_wait_s = 120;

pub const Fan = struct {
    links: [3]?Link = .{ null, null, null },

    pub fn open(io: std.Io, ip: [4]u8, port: u16, rank: u32, world: u32) !Fan {
        var f: Fan = .{};
        errdefer f.close();
        if (world > 4) return error.UnsupportedWorld;
        if (world < 2) return f;
        if (rank == 0) {
            for (1..world) |g| f.links[g - 1] = try Link.listen(ip, port + @as(u16, @intCast(g - 1)), link_wait_s);
        } else {
            f.links[0] = try Link.connect(io, ip, port + @as(u16, @intCast(rank - 1)), link_wait_s);
        }
        return f;
    }

    pub fn close(f: *Fan) void {
        for (&f.links) |*l| {
            if (l.*) |*x| x.close();
            l.* = null;
        }
    }

    /// NCCL over every link (CUDA builds; comm.zig Comm.fromFan): rank 0 makes the id and sends it to each follower.
    pub fn comm(f: *const Fan, rank: u32, world: u32) !@import("comm.zig").Comm {
        return @import("comm.zig").Comm.fromFan(f, rank, world);
    }
};

test "the split of each world" {
    try std.testing.expectEqual(@as(u32, 1), (try split(1, 2)).rank);
    const g3 = try split(3, 4);
    try std.testing.expectEqual(@as(u32, 1), g3.rank);
    try std.testing.expectEqual(@as(?u32, 1), g3.pair);
    try std.testing.expectError(error.UnsupportedWorld, split(0, 3));
}
