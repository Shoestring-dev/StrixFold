//! The stop marker: `stopping` in a run dir, which the launcher writes before stopping rank 0 so rank 1 reads a stop.
const std = @import("std");

/// The marker's file name, the launcher's (tools/dsv41-hip/first_light.sh down) and the engine's.
pub const name = "stopping";

/// Whether run dir `dir` holds the marker (an unreadable or overlong dir: no).
pub fn present(io: std.Io, dir: []const u8) bool {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name }) catch return false;
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

test "the marker is read only while the launcher's file is in the run dir" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try std.fmt.bufPrint(&buf, "{s}/{s}", .{ std.testing.TmpDir.parent_dir_path, &tmp.sub_path });
    try std.testing.expect(!present(io, dir));
    try std.testing.expect(!present(io, "/nonexistent/run"));
    (try tmp.dir.createFile(io, name, .{})).close(io);
    try std.testing.expect(present(io, dir));
    try tmp.dir.deleteFile(io, name);
    try std.testing.expect(!present(io, dir));
}
