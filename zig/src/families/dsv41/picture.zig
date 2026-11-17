//! A picture for the vision tower, as DeepSeek's MIT reference (image_processor.py) and the served lane make it:
//! the image decoded to RGB as PIL's convert("RGB") gives it (PNG here: every colour type, bit depth and interlace),
//! its grid (plan_image_grid: the patch grid of the ViT and the token grid after the 3x3 aligner), ImageOps.pad to the
//! grid's pixel size (PIL's 8-bit BICUBIC resample of the image fitted inside, 127 grey around it), then each value
//! (v / 255 - 0.5) / 0.5 rounded to bf16, in patch order [patch][channel][row][column] (the ViT's linear patch embed
//! reads each patch flattened that way). Every step is integer or IEEE float math in the reference's order, so the
//! patches are the lane's byte for byte (gate: tf-dsv41-picture against the recorded `decode` points).
const std = @import("std");
const Allocator = std.mem.Allocator;
const png = @import("picture_png.zig");
const resample = @import("picture_resample.zig");

pub const Error = png.Error;
pub const max_pixels = png.max_pixels;
pub const pixels = png.pixels;
pub const decodePng = png.decodePng;
pub const resize = resample.resize;
pub const pad = resample.pad;

/// The vision settings the grid depends on (config.json vision_config; DeepSeek-V4.1-Flash's values by default).
pub const Config = struct {
    patch: usize = 14,
    ratio: usize = 3, // the aligner's downsample
    max_tokens: usize = 1024, // an image span's tokens at most
    min_pixels: usize = 544 * 544,
};

/// Packed RGB rows, w * h * 3 bytes.
pub const Rgb = struct {
    w: usize,
    h: usize,
    px: []u8,

    pub fn deinit(r: *Rgb, a: Allocator) void {
        a.free(r.px);
        r.* = undefined;
    }
};

/// plan_image_grid's result: the padded pixel size (a whole number of patches each way), the ViT's patch grid and the
/// LLM's token grid.
pub const Grid = struct {
    best_h: usize,
    best_w: usize,
    n_vit_h: usize,
    n_vit_w: usize,
    n_llm_h: usize,
    n_llm_w: usize,

    /// num_image_tokens: [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END].
    pub fn spanTokens(g: Grid) usize {
        return g.n_llm_h * (g.n_llm_w + 1) + 2;
    }

    pub fn patches(g: Grid) usize {
        return g.n_vit_h * g.n_vit_w;
    }
};

/// The token types of an image span (image_token_types): IMAGE_START, then per row n_llm_w IMAGE and an
/// IMAGE_NEW_LINE, then IMAGE_END.
pub const Type = enum(u8) { start = 0, image = 1, newline = 2, end = 3 };

pub fn spanTypes(g: Grid, out: []Type) void {
    std.debug.assert(out.len == g.spanTokens());
    var i: usize = 0;
    out[i] = .start;
    i += 1;
    for (0..g.n_llm_h) |_| {
        for (0..g.n_llm_w) |_| {
            out[i] = .image;
            i += 1;
        }
        out[i] = .newline;
        i += 1;
    }
    out[i] = .end;
}

extern "c" fn pow(x: f64, y: f64) f64;
extern "c" fn sqrt(x: f64) f64;

fn ceilDiv(a: f64, b: f64) usize {
    return @intFromFloat(@ceil(a / b));
}

/// llm_grid: the token grid the aligner makes of a patch grid of this pixel size.
fn llmGrid(best_h: usize, best_w: usize, p: usize, r: usize) [2]usize {
    const fh: f64 = @floatFromInt(best_h / p);
    const fw: f64 = @floatFromInt(best_w / p);
    const fr: f64 = @floatFromInt(r);
    return .{ @intFromFloat(@ceil(fh / fr)), @intFromFloat(@ceil(fw / fr)) };
}

fn spanOf(n_llm_h: usize, n_llm_w: usize) usize {
    return n_llm_h * (n_llm_w + 1) + 2;
}

/// solve_resize_ratio: the largest aspect-preserving pixel size whose token grid fits in max_tokens.
fn solveResize(height: f64, width: f64, p: usize, r: usize, max_n: usize) [2]usize {
    const ratio = height / width;
    const max_w = sqrt(@as(f64, @floatFromInt(max_n - 2)) / ratio + 0.25) - 0.5;
    const max_h = max_w * ratio;
    const cell = p * r;
    if (max_w < 1.0) return .{ (max_n - 2) / 2 * cell, cell };
    if (max_h < 1.0) return .{ cell, (max_n - 3) * cell };
    const fcell: f64 = @floatFromInt(cell);
    const beta = @min(@floor(max_w) * fcell / width, @floor(max_h) * fcell / height);
    const fp: f64 = @floatFromInt(p);
    return .{ @as(usize, @intFromFloat(@floor(height * beta / fp))) * p, @as(usize, @intFromFloat(@floor(width * beta / fp))) * p };
}

/// plan_image_grid for an image of width x height pixels (max_wh_ratio unset, as this model's config has it).
pub fn plan(width0: usize, height0: usize, cfg: Config) Grid {
    const p = cfg.patch;
    var width: f64 = @floatFromInt(width0);
    var height: f64 = @floatFromInt(height0);
    const area = width0 * height0;
    if (area > 0 and area < cfg.min_pixels) {
        // ratio = (min_pixels / (width * height)) ** 0.5; width = int(width * ratio); height = int(height * ratio)
        const ratio = pow(@as(f64, @floatFromInt(cfg.min_pixels)) / @as(f64, @floatFromInt(area)), 0.5);
        width = @trunc(width * ratio);
        height = @trunc(height * ratio);
    }
    const fp: f64 = @floatFromInt(p);
    var best_w = ceilDiv(width, fp) * p;
    var best_h = ceilDiv(height, fp) * p;
    var g = llmGrid(best_h, best_w, p, cfg.ratio);
    if (spanOf(g[0], g[1]) > cfg.max_tokens) {
        const hw = solveResize(height, width, p, cfg.ratio, cfg.max_tokens);
        best_h = hw[0];
        best_w = hw[1];
        g = llmGrid(best_h, best_w, p, cfg.ratio);
    }
    return .{ .best_h = best_h, .best_w = best_w, .n_vit_h = best_h / p, .n_vit_w = best_w / p, .n_llm_h = g[0], .n_llm_w = g[1] };
}

/// f32 to bf16, round to nearest even (torch's .to(torch.bfloat16)).
pub fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b +% 0x7fff +% ((b >> 16) & 1)) >> 16);
}

/// The ViT's input of a padded image: n_vit_h * n_vit_w patches of [3][patch][patch] bf16, each value
/// (v / 255 - 0.5) / 0.5 in f32 (the reference's numpy / torch float32 steps).
pub fn patchify(a: Allocator, img: Rgb, p: usize) ![]u16 {
    const nh = img.h / p;
    const nw = img.w / p;
    const out = try a.alloc(u16, nh * nw * 3 * p * p);
    var lut: [256]u16 = undefined;
    for (0..256) |v| {
        const x = @as(f32, @floatFromInt(v)) / 255.0;
        lut[v] = bf16((x - 0.5) / 0.5);
    }
    for (0..nh) |ph| for (0..nw) |pw| {
        const base = (ph * nw + pw) * 3 * p * p;
        for (0..3) |c| for (0..p) |py| for (0..p) |px| {
            const y = ph * p + py;
            const x = pw * p + px;
            out[base + (c * p + py) * p + px] = lut[img.px[(y * img.w + x) * 3 + c]];
        };
    };
    return out;
}

/// A picture: its grid and its patches (owned).
pub const Picture = struct {
    grid: Grid,
    patches: []u16,

    pub fn deinit(p: *Picture, a: Allocator) void {
        a.free(p.patches);
        p.* = undefined;
    }
};

/// decode (load_image): bytes to the ViT's patches and the grids.
pub fn picture(a: Allocator, data: []const u8, cfg: Config) !Picture {
    var rgb = try decodePng(a, data);
    defer rgb.deinit(a);
    return fromRgb(a, rgb, cfg);
}

/// load_image after its decode: the decoded RGB planned, padded and cut into the ViT's patches.
pub fn fromRgb(a: Allocator, rgb: Rgb, cfg: Config) !Picture {
    const g = plan(rgb.w, rgb.h, cfg);
    var padded = try pad(a, rgb, g.best_w, g.best_h);
    defer padded.deinit(a);
    return .{ .grid = g, .patches = try patchify(a, padded, cfg.patch) };
}

test "grids of the vision gate's images as the reference plans them" {
    const cfg: Config = .{};
    // (w, h) -> best_w, best_h, n_vit_h, n_vit_w, n_llm_h, n_llm_w (py/vfix.py on the reference)
    const cases = [_][8]usize{
        .{ 560, 400, 644, 462, 33, 46, 11, 16 },
        .{ 420, 520, 490, 616, 44, 35, 15, 12 },
        .{ 400, 300, 630, 476, 34, 45, 12, 15 },
        .{ 120, 2000, 140, 2226, 159, 10, 53, 4 },
        .{ 640, 240, 896, 336, 24, 64, 8, 22 },
        .{ 16, 16, 546, 546, 39, 39, 13, 13 },
        .{ 3000, 90, 3150, 98, 7, 225, 3, 75 },
    };
    for (cases) |c| {
        const g = plan(c[0], c[1], cfg);
        try std.testing.expectEqual(c[2], g.best_w);
        try std.testing.expectEqual(c[3], g.best_h);
        try std.testing.expectEqual(c[4], g.n_vit_h);
        try std.testing.expectEqual(c[5], g.n_vit_w);
        try std.testing.expectEqual(c[6], g.n_llm_h);
        try std.testing.expectEqual(c[7], g.n_llm_w);
    }
}

test "bf16 rounds to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0xbf80), bf16(-1.0));
    try std.testing.expectEqual(@as(u16, 0x3f81), bf16(@bitCast(@as(u32, 0x3f80c000)))); // above half: up
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000)))); // half, even: down
}

test "a span's token types" {
    const g: Grid = .{ .best_h = 0, .best_w = 0, .n_vit_h = 0, .n_vit_w = 0, .n_llm_h = 2, .n_llm_w = 3 };
    var t: [10]Type = undefined;
    spanTypes(g, &t);
    try std.testing.expectEqualSlices(Type, &.{ .start, .image, .image, .image, .newline, .image, .image, .image, .newline, .end }, &t);
}
