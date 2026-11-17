//! PIL's 8-bit BICUBIC resize (Resample.c) and ImageOps.pad, for picture.zig.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Rgb = @import("picture.zig").Rgb;

/// Python's round() of a float: half to even.
fn pyRound(x: f64) i64 {
    const f = @floor(x);
    const d = x - f;
    var r = f;
    if (d > 0.5) r = f + 1 else if (d == 0.5) r = if (@mod(f, 2.0) == 0) f else f + 1;
    return @intFromFloat(r);
}

const precision_bits = 32 - 8 - 2;

fn bicubic(x0: f64) f64 {
    const a = -0.5;
    const x = @abs(x0);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

/// precompute_coeffs (Resample.c) with BICUBIC's support 2: each output pixel's first input pixel, its count and
/// fixed-point weights (normalize_coeffs_8bpc).
const Coeffs = struct {
    ksize: usize,
    bounds: []usize, // [out][2]: xmin, count
    k: []i32, // [out][ksize]

    fn init(a: Allocator, in_size: usize, in0: f64, in1: f64, out_size: usize) !Coeffs {
        const support0 = 2.0;
        const scale = (in1 - in0) / @as(f64, @floatFromInt(out_size));
        const filterscale = @max(scale, 1.0);
        const support = support0 * filterscale;
        const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
        const bounds = try a.alloc(usize, out_size * 2);
        errdefer a.free(bounds);
        const k = try a.alloc(i32, out_size * ksize);
        errdefer a.free(k);
        const kk = try a.alloc(f64, ksize);
        defer a.free(kk);
        for (0..out_size) |xx| {
            const center = in0 + (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
            const ss = 1.0 / filterscale;
            // (int)(center - support + 0.5), clamped: C truncation toward zero
            var xmin_i: i64 = @intFromFloat(@trunc(center - support + 0.5));
            if (xmin_i < 0) xmin_i = 0;
            var xmax_i: i64 = @intFromFloat(@trunc(center + support + 0.5));
            if (xmax_i > @as(i64, @intCast(in_size))) xmax_i = @intCast(in_size);
            xmax_i -= xmin_i;
            const xmin: usize = @intCast(xmin_i);
            const xmax: usize = @intCast(@max(xmax_i, 0));
            var ww: f64 = 0.0;
            for (0..xmax) |x| {
                const wv = bicubic((@as(f64, @floatFromInt(x + xmin)) - center + 0.5) * ss);
                kk[x] = wv;
                ww += wv;
            }
            for (0..xmax) |x| {
                if (ww != 0.0) kk[x] /= ww;
            }
            for (xmax..ksize) |x| kk[x] = 0;
            for (0..ksize) |x| {
                const v = kk[x] * @as(f64, @floatFromInt(@as(i64, 1) << precision_bits));
                // (int)(-0.5 + v) for negatives, (int)(0.5 + v) otherwise: truncation toward zero
                k[xx * ksize + x] = @intFromFloat(@trunc(if (kk[x] < 0) -0.5 + v else 0.5 + v));
            }
            bounds[2 * xx] = xmin;
            bounds[2 * xx + 1] = xmax;
        }
        return .{ .ksize = ksize, .bounds = bounds, .k = k };
    }

    fn deinit(c: *Coeffs, a: Allocator) void {
        a.free(c.bounds);
        a.free(c.k);
    }
};

/// clip8: the sum's integer part (an arithmetic shift) clamped to 0..255.
fn clip8(ss: i64) u8 {
    const v = ss >> precision_bits;
    return @intCast(std.math.clamp(v, 0, 255));
}

/// One BICUBIC pass over RGB rows: horizontal (input rows y0.. of the source, out w_out x h_out) or vertical.
fn passH(src: Rgb, y0: usize, out: *Rgb, c: Coeffs) void {
    for (0..out.h) |yy| {
        const row = src.px[(yy + y0) * src.w * 3 ..];
        for (0..out.w) |xx| {
            const xmin = c.bounds[2 * xx];
            const xmax = c.bounds[2 * xx + 1];
            const k = c.k[xx * c.ksize ..];
            var s: [3]i64 = @splat(@as(i64, 1) << (precision_bits - 1));
            for (0..xmax) |x| {
                for (0..3) |ch| s[ch] += @as(i64, row[(x + xmin) * 3 + ch]) * k[x];
            }
            for (0..3) |ch| out.px[(yy * out.w + xx) * 3 + ch] = clip8(s[ch]);
        }
    }
}

fn passV(src: Rgb, out: *Rgb, c: Coeffs) void {
    for (0..out.h) |yy| {
        const ymin = c.bounds[2 * yy];
        const ymax = c.bounds[2 * yy + 1];
        const k = c.k[yy * c.ksize ..];
        for (0..out.w) |xx| {
            var s: [3]i64 = @splat(@as(i64, 1) << (precision_bits - 1));
            for (0..ymax) |y| {
                for (0..3) |ch| s[ch] += @as(i64, src.px[((y + ymin) * src.w + xx) * 3 + ch]) * k[y];
            }
            for (0..3) |ch| out.px[(yy * out.w + xx) * 3 + ch] = clip8(s[ch]);
        }
    }
}

/// ImagingResampleInner with BICUBIC over the box (x0, y0, x1, y1) of the source: the horizontal pass first (on the
/// rows the vertical one reads), then the vertical one, each only when that side changes (Resample.c's need_horizontal
/// and need_vertical, compared as it compares them).
fn resampleBox(a: Allocator, src: Rgb, xsize: usize, ysize: usize, box: [4]f64) !Rgb {
    const need_h = xsize != src.w or box[0] != 0 or box[2] != @as(f64, @floatFromInt(xsize));
    const need_v = ysize != src.h or box[1] != 0 or box[3] != @as(f64, @floatFromInt(ysize));
    var ch = try Coeffs.init(a, src.w, box[0], box[2], xsize);
    defer ch.deinit(a);
    var cv = try Coeffs.init(a, src.h, box[1], box[3], ysize);
    defer cv.deinit(a);
    const ybox_first = cv.bounds[0];
    const ybox_last = cv.bounds[ysize * 2 - 2] + cv.bounds[ysize * 2 - 1];
    var cur = src;
    var owned: ?Rgb = null;
    errdefer if (owned) |*o| o.deinit(a);
    if (need_h) {
        for (0..ysize) |i| cv.bounds[i * 2] -= ybox_first;
        var tmp: Rgb = .{ .w = xsize, .h = ybox_last - ybox_first, .px = try a.alloc(u8, xsize * (ybox_last - ybox_first) * 3) };
        passH(src, ybox_first, &tmp, ch);
        owned = tmp;
        cur = tmp;
    }
    if (need_v) {
        var out: Rgb = .{ .w = cur.w, .h = ysize, .px = try a.alloc(u8, cur.w * ysize * 3) };
        passV(cur, &out, cv);
        if (owned) |*o| o.deinit(a);
        return out;
    }
    if (owned) |o| return o;
    return .{ .w = src.w, .h = src.h, .px = try a.dupe(u8, src.px) };
}

/// Image.resize(size, BICUBIC) of the whole source, a side of 0 refused as PIL does; past 100:1 tall, the height first.
pub fn resize(a: Allocator, src: Rgb, w: usize, h: usize) !Rgb {
    if (w == 0 or h == 0) return error.EmptySide;
    if (w == src.w and h == src.h) return .{ .w = w, .h = h, .px = try a.dupe(u8, src.px) };
    const fw: f64 = @floatFromInt(src.w);
    const fh: f64 = @floatFromInt(src.h);
    if (src.h > src.w * 100 and h < src.h) {
        var tall = try resampleBox(a, src, src.w, h, .{ 0, 0, fw, fh });
        defer tall.deinit(a);
        return resampleBox(a, tall, w, h, .{ 0, 0, fw, @floatFromInt(h) });
    }
    return resampleBox(a, src, w, h, .{ 0, 0, fw, fh });
}

/// ImageOps.pad(image, (w, h), color=(127, 127, 127)): contain() (the aspect kept, Python's round of the scaled side),
/// then pasted centred (Python's round of half the gap) on a grey canvas.
pub fn pad(a: Allocator, src: Rgb, w: usize, h: usize) !Rgb {
    const im_ratio = @as(f64, @floatFromInt(src.w)) / @as(f64, @floatFromInt(src.h));
    const dest_ratio = @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(h));
    var cw = w;
    var chh = h;
    if (im_ratio != dest_ratio) {
        if (im_ratio > dest_ratio) {
            const nh: usize = @intCast(pyRound(@as(f64, @floatFromInt(src.h)) / @as(f64, @floatFromInt(src.w)) * @as(f64, @floatFromInt(w))));
            if (nh != h) chh = nh;
        } else {
            const nw: usize = @intCast(pyRound(@as(f64, @floatFromInt(src.w)) / @as(f64, @floatFromInt(src.h)) * @as(f64, @floatFromInt(h))));
            if (nw != w) cw = nw;
        }
    }
    var fit = try resize(a, src, cw, chh);
    if (fit.w == w and fit.h == h) return fit;
    defer fit.deinit(a);
    const px = try a.alloc(u8, w * h * 3);
    @memset(px, 127);
    var x0: usize = 0;
    var y0: usize = 0;
    if (fit.w != w) {
        x0 = @intCast(pyRound(@as(f64, @floatFromInt(w - fit.w)) * 0.5));
    } else {
        y0 = @intCast(pyRound(@as(f64, @floatFromInt(h - fit.h)) * 0.5));
    }
    for (0..fit.h) |y| {
        @memcpy(px[((y + y0) * w + x0) * 3 ..][0 .. fit.w * 3], fit.px[y * fit.w * 3 ..][0 .. fit.w * 3]);
    }
    return .{ .w = w, .h = h, .px = px };
}

test "Python's round: half to even" {
    try std.testing.expectEqual(@as(i64, 2), pyRound(2.5));
    try std.testing.expectEqual(@as(i64, 4), pyRound(3.5));
    try std.testing.expectEqual(@as(i64, 3), pyRound(2.6));
    try std.testing.expectEqual(@as(i64, 0), pyRound(0.5));
}
