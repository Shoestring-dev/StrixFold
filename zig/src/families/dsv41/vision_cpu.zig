//! On-demand CPU vision; text requests start no encoder and allocate no tower weights.
const std = @import("std");
const lanes = @import("lanes");
const picture = @import("picture.zig");
const model = @import("model.zig");
const pil = @import("pil.zig");
const image_cache = @import("vision_cache.zig");

/// A prepared payload is a six-u32 little-endian grid followed by sixteen-byte-aligned BF16 patches.
const header_bytes = 32;

pub const Vision = struct {
    gpa: std.mem.Allocator,
    m: *model.Model,
    cfg: picture.Config = .{},
    io: std.Io,
    python: []const u8,
    weights: []const u8,
    digest: []const u8,
    hip_code: ?[]const u8,
    hip_digest: ?[]const u8,
    cache: image_cache.Cache,
    pil: ?*pil.Pil = null, // the formats picture.zig does not decode (null: no python3 with Pillow here)

    /// Register the optional tower without loading its weights or starting Torch.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, m: *model.Model, model_dir: []const u8, kit_dir: []const u8) !*Vision {
        _ = model_dir;
        _ = kit_dir;
        const python = env("TF_DS_VISION_PYTHON") orelse return error.NoVisionCpuConfig;
        const weights = env("TF_DS_VISION_WEIGHTS") orelse return error.NoVisionCpuConfig;
        const digest = env("TF_DS_VISION_SHA256") orelse return error.NoVisionCpuConfig;
        if (digest.len != 64 or python.len == 0 or python[0] != '/' or weights.len == 0 or weights[0] != '/') return error.NoVisionCpuConfig;
        if (m.w.layers.len == 0 or m.w.layers[0].gate_b_vl == 0) return error.NoVisionKit;
        const code = env("TF_DS_VISION_HIP_CODE");
        const code_digest = env("TF_DS_VISION_HIP_SHA256");
        const cache_mib = std.fmt.parseInt(usize, env("TF_DS_VISION_CACHE_MIB") orelse "0", 10) catch
            return error.NoVisionCpuConfig;
        if ((code == null) != (code_digest == null) or cache_mib > 512) return error.NoVisionCpuConfig;
        if (code) |path| if (!std.fs.path.isAbsolute(path) or code_digest.?.len != 64) return error.NoVisionCpuConfig;
        const v = try gpa.create(Vision);
        v.* = .{ .gpa = gpa, .m = m, .io = io, .python = python, .weights = weights, .digest = digest,
            .hip_code = code, .hip_digest = code_digest,
            .cache = .{ .allocator = gpa, .budget = cache_mib << 20 } };
        // PNG decoding is native; other formats use the existing bounded Pillow decoder if installed.
        v.pil = pil.Pil.start(gpa, io) catch null;
        return v;
    }

    fn env(name: [*:0]const u8) ?[]const u8 {
        return if (std.c.getenv(name)) |x| std.mem.span(x) else null;
    }

    pub fn close(v: *Vision) void {
        if (v.pil) |p| p.stop(v.gpa);
        v.cache.deinit();
        v.gpa.destroy(v);
    }

    /// The server's view: the placeholder the template renders for each image and the picture it prepares.
    pub fn hook(v: *Vision) lanes.stream.Vision {
        return .{ .ctx = v, .placeholder = placeholder, .token = v.m.cfg.image_token.?,
            .cache_images = v.cache.budget > 0, .prepare = prepareFn };
    }

    pub const placeholder = "<\u{ff5c}deepseek_image\u{ff5c}>";

    fn prepareFn(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, problem: *[]const u8) lanes.stream.Vision.PrepareError!lanes.stream.Vision.Prepared {
        const v: *Vision = @ptrCast(@alignCast(ctx));
        var rgb = try v.decode(a, bytes, problem);
        defer rgb.deinit(a);
        var pic = picture.fromRgb(a, rgb, v.cfg) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            problem.* = "image input: the image could not be prepared";
            return error.BadImage;
        };
        defer pic.deinit(a);
        const g = pic.grid;
        const payload = try a.alignedAlloc(u8, .@"16", header_bytes + pic.patches.len * 2);
        @memset(payload[0..header_bytes], 0);
        for ([_]usize{ g.best_h, g.best_w, g.n_vit_h, g.n_vit_w, g.n_llm_h, g.n_llm_w }, 0..) |x, i| std.mem.writeInt(u32, payload[i * 4 ..][0..4], @intCast(x), .little);
        @memcpy(payload[header_bytes..], std.mem.sliceAsBytes(pic.patches));
        return .{ .tokens = @intCast(g.spanTokens()), .payload = payload };
    }

    /// Decode pixels through the native PNG decoder or the bounded Pillow helper.
    fn decode(v: *Vision, a: std.mem.Allocator, bytes: []const u8, problem: *[]const u8) lanes.stream.Vision.PrepareError!picture.Rgb {
        if (picture.decodePng(a, bytes)) |rgb| return rgb else |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.TooLarge => {
                problem.* = try std.fmt.allocPrint(a, "Image size ({d} pixels) exceeds limit of {d} pixels, could be decompression bomb DOS attack.", .{ picture.pixels(bytes) orelse 0, picture.max_pixels });
                return error.BadImage;
            },
            else => {}, // not a PNG this decoder reads: Pillow's
        }
        const p = v.pil orelse {
            problem.* = "image input: cannot identify image file (no Pillow on this server for formats other than PNG)";
            return error.BadImage;
        };
        const got = p.decode(a, bytes) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            std.debug.print("{{\"vision\": \"Pillow's child failed ({s})\"}}\n", .{@errorName(err)});
            problem.* = "image input: the image decoder failed";
            return error.BadImage;
        };
        switch (got) {
            .rgb => |rgb| return rgb,
            .refused => |m| {
                problem.* = m;
                return error.BadImage;
            },
        }
    }

    fn checkPayload(v: *Vision, payload: []const u8, tokens: u32) !void {
        if (payload.len < header_bytes) return error.BadImagePayload;
        var x: [6]usize = undefined;
        for (&x, 0..) |*f, i| f.* = std.mem.readInt(u32, payload[i * 4 ..][0..4], .little);
        const g: picture.Grid = .{ .best_h = x[0], .best_w = x[1], .n_vit_h = x[2], .n_vit_w = x[3], .n_llm_h = x[4], .n_llm_w = x[5] };
        if (g.n_vit_h == 0 or g.n_vit_w == 0 or g.n_vit_h > 9216 or g.n_vit_w > 9216 or
            g.patches() > 9216 or g.best_h != g.n_vit_h * 14 or g.best_w != g.n_vit_w * 14 or
            g.n_llm_h != (g.n_vit_h + 2) / 3 or g.n_llm_w != (g.n_vit_w + 2) / 3)
            return error.BadImagePayload;
        if (payload.len != header_bytes + g.patches() * 588 * 2 or
            g.spanTokens() != tokens or tokens > v.cfg.max_tokens) return error.BadImagePayload;
    }

    fn start(v: *Vision, payload: []const u8) !std.process.Child {
        const cpu = @embedFile("vision_cpu.py");
        const hip = @embedFile("vision_hip.py") ++ "\n" ++ cpu;
        const argv: []const []const u8 = if (v.hip_code) |code|
            &.{ v.python, "-I", "-c", hip, v.weights, v.digest, code, v.hip_digest.? }
        else
            &.{ v.python, "-I", "-c", cpu, v.weights, v.digest };
        var child = try std.process.spawn(v.io, .{
            .argv = argv,
            .stdin = .pipe, .stdout = .pipe, .stderr = .inherit,
        });
        errdefer child.kill(v.io);
        try child.stdin.?.writeStreamingAll(v.io, payload);
        child.stdin.?.close(v.io);
        child.stdin = null;
        return child;
    }

    fn finish(v: *Vision, owned: *?std.process.Child, out: []u8, tokens: u32, cancel: ?lanes.stream.CancelCheck) !void {
        const child = if (owned.*) |*p| p else unreachable;
        const fd = child.stdout.?.handle;
        var head: [16]u8 = undefined;
        try readReply(fd, &head, cancel);
        const rows_n = std.mem.readInt(u32, head[4..8], .little);
        const bytes = std.mem.readInt(u64, head[8..16], .little);
        if (!std.mem.eql(u8, head[0..4], "VCPU") or rows_n != tokens or bytes != tokens * 5120 * 2) return error.BadImageRows;
        if (bytes != out.len) return error.BadImageRows;
        try readReply(fd, out, cancel);
        try waitForReply(fd, cancel);
        var trailing: [1]u8 = undefined;
        if (std.c.read(fd, &trailing, trailing.len) != 0) return error.BadImageRows;
        const term = try child.wait(v.io);
        owned.* = null;
        if (term != .exited or term.exited != 0) return error.VisionCpuFailed;
    }

    /// Encode uncached images on the serving device and retain bounded rows in prompt order.
    pub fn rowsBatch(v: *Vision, images: []const lanes.stream.Image, gpa: std.mem.Allocator, out: *std.ArrayList(u8), request: *const lanes.Stream) !void {
        var hits: usize = 0;
        var jobs: usize = 0;
        for (images) |img| {
            if (request.isCancelled()) return error.Cancelled;
            const size = @as(usize, img.tokens) * 5120 * 2;
            if (img.rows) {
                if (img.bytes.len != size) return error.BadImageRows;
                try out.appendSlice(gpa, img.bytes);
                continue;
            }
            try v.checkPayload(img.bytes, img.tokens);
            const digest = image_cache.key(img.bytes);
            if (v.cache.get(digest)) |cached| {
                if (cached.len != size) return error.BadImageRows;
                try out.appendSlice(gpa, cached);
                hits += 1;
                continue;
            }
            var child: ?std.process.Child = try v.start(img.bytes);
            defer if (child) |*owned| owned.kill(v.io);
            const row_offset = out.items.len;
            try out.resize(gpa, row_offset + size);
            try v.finish(&child, out.items[row_offset..], img.tokens, request.cancel_check);
            if (request.isCancelled()) return error.Cancelled;
            v.cache.put(digest, out.items[row_offset..]);
            jobs += 1;
        }
        std.log.info("vision rows: {d} worker jobs, {d} image cache hits", .{ jobs, hits });
    }

};

fn readReply(fd: std.c.fd_t, dest: []u8, cancel: ?lanes.stream.CancelCheck) !void {
    var at: usize = 0;
    while (at < dest.len) {
        try waitForReply(fd, cancel);
        const n = std.c.read(fd, dest[at..].ptr, dest.len - at);
        if (n <= 0) return error.VisionCpuPipe;
        at += @intCast(n);
    }
}

/// The owning thread polls cancellation and kills/reaps its child without PID lookups.
fn waitForReply(fd: std.c.fd_t, cancel: ?lanes.stream.CancelCheck) !void {
    var fds = [_]std.c.pollfd{.{ .fd = fd, .events = std.c.POLL.IN, .revents = 0 }};
    while (true) {
        if (cancel) |c| if (c.check(c.ptr)) return error.Cancelled;
        const count = std.c.poll(&fds, 1, 100);
        if (count < 0 or fds[0].revents & std.c.POLL.NVAL != 0) return error.VisionCpuPipe;
        if (count > 0) return;
    }
}

test "a cancelled image does not wait for the encoder" {
    var marker: u8 = 0;
    const check = struct {
        fn call(_: *anyopaque) bool { return true; }
    }.call;
    try std.testing.expectError(error.Cancelled, waitForReply(-1, .{ .ptr = &marker, .check = check }));
}

test "an invalid encoder pipe fails closed" {
    try std.testing.expectError(error.VisionCpuPipe, waitForReply(1000000, null));
}

test "a payload's grid reads back" {
    var buf: [header_bytes]u8 = undefined;
    for ([_]u32{ 462, 644, 33, 46, 11, 16 }, 0..) |x, i| std.mem.writeInt(u32, buf[i * 4 ..][0..4], x, .little);
    try std.testing.expectEqual(@as(u32, 11), std.mem.readInt(u32, buf[16..20], .little));
    const g: picture.Grid = .{ .best_h = 462, .best_w = 644, .n_vit_h = 33, .n_vit_w = 46, .n_llm_h = 11, .n_llm_w = 16 };
    try std.testing.expectEqual(@as(usize, 189), g.spanTokens());
}
