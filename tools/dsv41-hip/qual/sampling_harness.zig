//! sampling_harness.zig: drives the engine's own target sampler (zig/src/families/dsv41/sampling.zig, imported as the
//! module "sampling") on generated rows for qual/sampling_check.py (gate M4.sampling-contract). Rows, seeds and
//! positions come from a splitmix64 stream that the checker regenerates bit for bit; every value is an integer over
//! 128 or over 2 (exact in f32, the second with many ties). Two modes, written to stderr:
//!   cases: "C <case> <token>" for every (width, temperature, top_k, top_p, min_p) setting, 5 rows each
//!   draws: "D <dist> <token> <count>" for 12 small distributions, a fixed seed and positions 0 .. 19999
const std = @import("std");
const sampling = @import("sampling");

fn mix(x0: u64) u64 {
    var x = x0;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

const Gen = struct {
    s: u64,
    fn next(g: *Gen) u64 {
        g.s +%= 0x9E3779B97F4A7C15;
        return mix(g.s);
    }
    fn value(g: *Gen, coarse: bool) f32 {
        const q: i64 = @as(i64, @intCast(g.next() >> 52)) - 2048;
        return @as(f32, @floatFromInt(q)) / (if (coarse) @as(f32, 2.0) else @as(f32, 128.0));
    }
};

const temps = [_]f64{ 0.0, 0.3, 0.7, 1.0, 1.5 };
const topks = [_]usize{ 1, 5, 20, 56 };
const topps = [_]f64{ 1.0, 0.95, 0.5, 0.1 };
const minps = [_]f64{ 0.0, 0.05, 0.3 };
const widths = [_]usize{ 1, 2, 7, 30, 300 };

pub fn main() !void {
    var g: Gen = .{ .s = 0x5EED_0F_0A11 };
    var row: [300]f32 = undefined;
    var case: u64 = 0;
    for (widths) |n| for (temps) |t| for (topks) |k| for (topps) |p| for (minps) |m| for (0..5) |r| {
        const coarse = r % 2 == 1;
        for (row[0..n]) |*v| v.* = g.value(coarse);
        const seed = g.next();
        const pos = g.next() % 100000;
        const tok = try sampling.sampleRow(row[0..n], pos, .{ .seed = seed, .temperature = t, .top_k = k, .top_p = p, .min_p = m });
        std.debug.print("C {d} {d}\n", .{ case, tok });
        case += 1;
    };
    // draws: 12 small distributions (width 4..16), each sampled at positions 0 .. 19999 with one seed
    var counts: [16]u64 = undefined;
    for (0..12) |d| {
        const n = 4 + (d % 4) * 4;
        for (row[0..n]) |*v| v.* = g.value(false) / 4.0; // logits in [-4, 4): every token likely enough to count
        const seed = g.next();
        const st = sampling.Sampling{ .seed = seed, .temperature = temps[1 + d % 4], .top_k = if (d % 3 == 0) n else 5, .top_p = if (d % 2 == 0) 1.0 else 0.9, .min_p = if (d % 5 == 4) 0.05 else 0.0 };
        @memset(counts[0..n], 0);
        for (0..20000) |pos| {
            const tok = try sampling.sampleRow(row[0..n], pos, st);
            counts[@intCast(tok)] += 1;
        }
        for (0..n) |i| std.debug.print("D {d} {d} {d}\n", .{ d, i, counts[i] });
    }
}
