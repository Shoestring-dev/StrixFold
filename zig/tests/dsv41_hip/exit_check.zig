//! ``tf-dsv41-exit-check LIB [--return]``: opens and closes LIB, then leaves with 3 by leave() (--return: libc exit()).
const std = @import("std");
const leave = @import("leave");

const status: u8 = 3;

pub fn main(init: std.process.Init) !u8 {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 2 or argv.len > 3 or (argv.len == 3 and !std.mem.eql(u8, argv[2], "--return"))) {
        std.debug.print("usage: tf-dsv41-exit-check LIB [--return]\n", .{});
        return 2;
    }
    var lib = std.DynLib.open(argv[1]) catch |e| {
        std.debug.print("tf-dsv41-exit-check: cannot open {s} ({s})\n", .{ argv[1], @errorName(e) });
        return 2;
    };
    lib.close();
    // C stdio, which a pipe buffers until exit(): leave() has to flush it
    _ = std.c.printf("exit-check: the library was opened and closed; leaving with %d\n", @as(c_int, status));
    if (argv.len == 3) return status;
    leave.leave(status);
}
