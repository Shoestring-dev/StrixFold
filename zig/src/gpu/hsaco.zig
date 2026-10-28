//! What an AMD GPU code object says about itself: its ELF machine flags and, from the NT_AMDGPU_METADATA note
//! (MessagePack), each kernel's symbol, argument layout, workgroup bound, wavefront width, LDS, scratch and registers.
//! The kit tool records these in the manifest and the loader re-reads them, so the kernel ABI is checked against what
//! the compiler emitted, never against a name.

const std = @import("std");
const msgpack = @import("msgpack.zig");

pub const Error = error{ NotElf, NotAmdgpu, NoMetadata, BadMetadata } || msgpack.Error;

pub const em_amdgpu = 224;

/// The EF_AMDGPU_MACH_* value of e_flags for each arch this port builds. Only values read from a real code object are
/// listed (gfx1151: 0x4a, hipcc 7.15 on node1); others are refused.
pub fn machOf(arch: []const u8) ?u32 {
    const known = [_]struct { []const u8, u32 }{.{ "gfx1151", 0x4a }};
    for (known) |k| if (std.mem.eql(u8, k[0], arch)) return k[1];
    return null;
}
const nt_amdgpu_metadata = 32;
const sht_note = 7;

pub const Arg = struct {
    offset: u32,
    size: u32,
    /// The metadata's .value_kind: by_value, global_buffer, dynamic_shared_pointer, hidden_block_count_x, ...
    kind: []const u8,

    pub fn hidden(a: Arg) bool {
        return std.mem.startsWith(u8, a.kind, "hidden_");
    }
};

pub const Kernel = struct {
    name: []const u8,
    symbol: []const u8,
    args: []const Arg,
    kernarg_size: u32,
    wave: u32,
    max_flat_workgroup_size: u32,
    lds_static: u32,
    scratch: u32,
    sgpr: u32,
    vgpr: u32,
    dynamic_stack: bool,

    /// The arguments the caller passes (HIP fills the hidden ones).
    pub fn explicit(k: Kernel) []const Arg {
        var n: usize = 0;
        while (n < k.args.len and !k.args[n].hidden()) n += 1;
        return k.args[0..n];
    }
};

pub const CodeObject = struct {
    e_flags: u32,
    target: []const u8,
    kernels: []const Kernel,

    pub fn mach(c: CodeObject) u32 {
        return c.e_flags & 0xff;
    }

    pub fn find(c: CodeObject, name: []const u8) ?Kernel {
        for (c.kernels) |k| if (std.mem.eql(u8, k.name, name)) return k;
        return null;
    }
};

/// Parses `bytes` (a raw code-object ELF); strings in the result point into `bytes` or the arena.
pub fn parse(a: std.mem.Allocator, bytes: []const u8) Error!CodeObject {
    if (bytes.len < 64 or !std.mem.eql(u8, bytes[0..4], "\x7fELF") or bytes[4] != 2 or bytes[5] != 1) return error.NotElf;
    if (std.mem.readInt(u16, bytes[18..20], .little) != em_amdgpu) return error.NotAmdgpu;
    const e_flags = std.mem.readInt(u32, bytes[48..52], .little);
    const shoff = std.mem.readInt(u64, bytes[0x28..][0..8], .little);
    const shentsize = std.mem.readInt(u16, bytes[0x3a..][0..2], .little);
    const shnum = std.mem.readInt(u16, bytes[0x3c..][0..2], .little);
    for (0..shnum) |i| {
        const h = shoff + i * shentsize;
        if (h + 64 > bytes.len) return error.NotElf;
        if (std.mem.readInt(u32, bytes[h + 4 ..][0..4], .little) != sht_note) continue;
        const off = std.mem.readInt(u64, bytes[h + 24 ..][0..8], .little);
        const size = std.mem.readInt(u64, bytes[h + 32 ..][0..8], .little);
        const al = @max(4, std.mem.readInt(u64, bytes[h + 48 ..][0..8], .little));
        if (off > bytes.len or size > bytes.len - off) return error.NotElf;
        if (try metadataNote(bytes[off..][0..size], al)) |desc| return interpret(a, e_flags, try msgpack.decode(a, desc));
    }
    return error.NoMetadata;
}

/// The NT_AMDGPU_METADATA descriptor of a note section, if it holds one.
fn metadataNote(sec: []const u8, al: u64) Error!?[]const u8 {
    var at: usize = 0;
    while (at + 12 <= sec.len) {
        const namesz = std.mem.readInt(u32, sec[at..][0..4], .little);
        const descsz = std.mem.readInt(u32, sec[at + 4 ..][0..4], .little);
        const typ = std.mem.readInt(u32, sec[at + 8 ..][0..4], .little);
        const name_at = at + 12;
        const desc_at = name_at + std.mem.alignForward(usize, namesz, al);
        const next = desc_at + std.mem.alignForward(usize, descsz, al);
        if (desc_at > sec.len or descsz > sec.len - desc_at) return error.NotElf;
        const name = sec[name_at..][0..namesz];
        if (typ == nt_amdgpu_metadata and std.mem.eql(u8, std.mem.sliceTo(name, 0), "AMDGPU")) return sec[desc_at..][0..descsz];
        at = next;
    }
    return null;
}

fn num(v: msgpack.Value, key: []const u8) Error!u32 {
    const x = (v.get(key) orelse return error.BadMetadata).unsigned() orelse return error.BadMetadata;
    return std.math.cast(u32, x) orelse error.BadMetadata;
}

fn interpret(a: std.mem.Allocator, e_flags: u32, root: msgpack.Value) Error!CodeObject {
    const target = (root.get("amdhsa.target") orelse return error.BadMetadata).string() orelse return error.BadMetadata;
    const list = root.get("amdhsa.kernels") orelse return error.BadMetadata;
    if (list != .array) return error.BadMetadata;
    const kernels = try a.alloc(Kernel, list.array.len);
    for (list.array, kernels) |kv, *k| {
        const args_v = kv.get(".args");
        const n = if (args_v) |x| (if (x == .array) x.array.len else return error.BadMetadata) else 0;
        const args = try a.alloc(Arg, n);
        for (args, 0..) |*arg, i| {
            const av = args_v.?.array[i];
            arg.* = .{ .offset = try num(av, ".offset"), .size = try num(av, ".size"), .kind = (av.get(".value_kind") orelse return error.BadMetadata).string() orelse return error.BadMetadata };
        }
        k.* = .{
            .name = (kv.get(".name") orelse return error.BadMetadata).string() orelse return error.BadMetadata,
            .symbol = (kv.get(".symbol") orelse return error.BadMetadata).string() orelse return error.BadMetadata,
            .args = args,
            .kernarg_size = try num(kv, ".kernarg_segment_size"),
            .wave = try num(kv, ".wavefront_size"),
            .max_flat_workgroup_size = try num(kv, ".max_flat_workgroup_size"),
            .lds_static = try num(kv, ".group_segment_fixed_size"),
            .scratch = try num(kv, ".private_segment_fixed_size"),
            .sgpr = try num(kv, ".sgpr_count"),
            .vgpr = try num(kv, ".vgpr_count"),
            .dynamic_stack = if (kv.get(".uses_dynamic_stack")) |x| (x == .boolean and x.boolean) else false,
        };
    }
    return .{ .e_flags = e_flags, .target = target, .kernels = kernels };
}

test "a non-ELF or non-AMDGPU image is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.NotElf, parse(arena.allocator(), "__CLANG_OFFLOAD_BUNDLE__" ++ @as([64]u8, @splat(0))));
    var x86: [64]u8 = @splat(0);
    @memcpy(x86[0..4], "\x7fELF");
    x86[4] = 2;
    x86[5] = 1;
    std.mem.writeInt(u16, x86[18..20], 62, .little);
    try std.testing.expectError(error.NotAmdgpu, parse(arena.allocator(), &x86));
}
