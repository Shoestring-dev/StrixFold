//! The server's exit without libc exit(), for an engine whose runtime cannot run its finalisers after the close.
const std = @import("std");

extern "c" fn fflush(stream: ?*anyopaque) c_int;

/// Ends the process with `code` and no exit handlers; C stdio is flushed first, as exit() would have flushed it.
pub fn leave(code: u8) noreturn {
    _ = fflush(null);
    std.c._exit(code);
}
