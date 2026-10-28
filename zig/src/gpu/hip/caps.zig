//! What an RDNA GPU can do, from its exact gfx name (after ashhart/TensorFold d46d17a zig/src/hip/caps.zig, plus FP16
//! WMMA), and the device facts a family reads at run time. Kernels and families test these, never the name.
const std = @import("std");

pub const Matrix = enum { none, wmma11, wmma12 };
pub const Generation = enum { rdna2, rdna3, rdna3_5, rdna4 };

/// The instruction set of a gfx name.
pub const Isa = struct {
    wave: u8,
    dot2_f16: bool,
    dot2_bf16: bool,
    sdot4: bool,
    sdot8: bool,
    matrix: Matrix,
    /// v_wmma_f32_16x16x16_f16 (FP16 inputs, FP32 accumulate): what EXL3's reconstructed FP16 tiles need.
    wmma_f16: bool,
    /// v_wmma_f32_16x16x16_bf16.
    wmma_bf16: bool,
    generation: Generation,

    /// The ISA of `name` ("gfx1151", "gfx1151:xnack-" ...), or null for a GPU outside the table.
    pub fn of(name: []const u8) ?Isa {
        const end = std.mem.indexOfScalar(u8, name, ':') orelse name.len;
        for (table) |row| {
            for (row.names) |n| if (std.mem.eql(u8, n, name[0..end])) return row.isa;
        }
        return null;
    }
};

const Row = struct { names: []const []const u8, isa: Isa };

/// One row a generation; every part of it shares the instructions the kernels use.
const table = [_]Row{
    .{ .names = &.{ "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036" }, .isa = .{ .wave = 32, .dot2_f16 = true, .dot2_bf16 = false, .sdot4 = true, .sdot8 = true, .matrix = .none, .wmma_f16 = false, .wmma_bf16 = false, .generation = .rdna2 } },
    .{ .names = &.{ "gfx1100", "gfx1101", "gfx1102", "gfx1103" }, .isa = .{ .wave = 32, .dot2_f16 = true, .dot2_bf16 = true, .sdot4 = true, .sdot8 = true, .matrix = .wmma11, .wmma_f16 = true, .wmma_bf16 = true, .generation = .rdna3 } },
    .{ .names = &.{ "gfx1150", "gfx1151", "gfx1152", "gfx1153" }, .isa = .{ .wave = 32, .dot2_f16 = true, .dot2_bf16 = true, .sdot4 = true, .sdot8 = true, .matrix = .wmma11, .wmma_f16 = true, .wmma_bf16 = true, .generation = .rdna3_5 } },
    .{ .names = &.{ "gfx1200", "gfx1201" }, .isa = .{ .wave = 32, .dot2_f16 = true, .dot2_bf16 = true, .sdot4 = true, .sdot8 = true, .matrix = .wmma12, .wmma_f16 = true, .wmma_bf16 = true, .generation = .rdna4 } },
};

/// The device as the family sees it: its exact name, ISA and the runtime's numbers. `pdl`, `clusters` and
/// `cooperative` are CUDA launch features with no HIP equivalent here; families read them instead of the backend tag.
pub const Caps = struct {
    arch: [32]u8 = @splat(0),
    arch_len: u8 = 0,
    isa: Isa,
    cus: u32,
    lds_per_block: u32,
    integrated: bool,
    wall_clock_hz: u64,
    pdl: bool = false,
    clusters: bool = false,
    cooperative: bool = false,
    graphs: bool = true,
    host_mapped: bool = true,

    pub fn archName(c: *const Caps) []const u8 {
        return c.arch[0..c.arch_len];
    }
};

test "each generation gets its instructions, FP16 WMMA from RDNA3 on" {
    const rdna2 = Isa.of("gfx1030").?;
    try std.testing.expect(rdna2.dot2_f16 and !rdna2.dot2_bf16 and rdna2.matrix == .none and !rdna2.wmma_f16);
    const strix = Isa.of("gfx1151:xnack-").?;
    try std.testing.expect(strix.wmma_f16 and strix.wmma_bf16 and strix.matrix == .wmma11 and strix.generation == .rdna3_5);
    for (table) |row| try std.testing.expectEqual(@as(u8, 32), row.isa.wave);
}

test "names outside the table are refused" {
    for ([_][]const u8{ "gfx1010", "gfx1037", "gfx1104", "gfx1202", "gfx906", "gfx90a", "gfx", "", "sm_121", "gfx11510" }) |name|
        try std.testing.expect(Isa.of(name) == null);
}
