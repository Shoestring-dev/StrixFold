//! Structured output on the host: response_format (json_object, json_schema), guided_json/regex/choice and a required
//! DeepSeek-V4.1 DSML tool call as byte grammars (program.zig, machine.zig), a tokenizer's vocabulary as bytes
//! (vocab.zig), and the per-reply Constraint a backend's draws consult (constraint.zig): every committed token keeps
//! the reply inside its grammar. The machine runs on the host of the rank that draws; nothing here touches a device.
const std = @import("std");

pub const trie = @import("trie.zig");
pub const regex = @import("regex.zig");
pub const program = @import("program.zig");
pub const machine = @import("machine.zig");
pub const vocab = @import("vocab.zig");
pub const constraint = @import("constraint.zig");

pub const Structure = program.Structure;
pub const Diag = program.Diag;
pub const Program = program.Program;
pub const Vocab = vocab.Vocab;
pub const Constraint = constraint.Constraint;

/// Why `s` cannot be served (its message lives in `diag`), or null when it compiles.
pub const check = program.check;

test {
    std.testing.refAllDecls(@This());
    _ = @import("grammar_test.zig");
}
