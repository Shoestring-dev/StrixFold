//! sf-disk's identity for dsv41: the namespace text (what a kept state's bytes depend on) and the chain's seed text.
const std = @import("std");
const gpu = @import("gpu");
const core = @import("core");
const model = @import("model.zig");
const disk = @import("disk.zig");
const pd = core.prompt_disk;
const Model = model.Model;
const block_tokens = disk.block_tokens;
const Plane = disk.Plane;
const envVar = disk.envVar;

/// A rank's namespace text (pd.namespaceOf names its directory): what its states' bytes depend on, less conf=/disk=.
pub fn identityText(a: std.mem.Allocator, parts: Parts) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    const w = struct {
        fn f(o: *std.ArrayList(u8), al: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
            const s = try std.fmt.allocPrint(al, fmt ++ "\n", args);
            defer al.free(s);
            try o.appendSlice(al, s);
        }
    }.f;
    try w(&out, a, "format={d}", .{pd.version});
    try w(&out, a, "rank={d}", .{parts.rank});
    try w(&out, a, "world={d}", .{parts.world});
    try w(&out, a, "deploy={s}", .{parts.deploy});
    try w(&out, a, "bundle={s}", .{parts.bundle});
    try w(&out, a, "kit_sha256={s}", .{parts.kit_sha256});
    try w(&out, a, "hip_runtime={d}", .{parts.hip_runtime});
    try w(&out, a, "arch={s}", .{parts.arch});
    var it = std.mem.tokenizeScalar(u8, parts.plan, ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, "conf=") or std.mem.startsWith(u8, tok, "disk=")) continue;
        try w(&out, a, "plan.{s}", .{tok});
    }
    try w(&out, a, "engram={s}", .{parts.engram});
    try w(&out, a, "token_map_sha256={s}", .{parts.token_map});
    try w(&out, a, "ds_kit={s}", .{parts.ds_kit});
    try w(&out, a, "window={d}", .{parts.window});
    try w(&out, a, "snap_bytes={d}", .{parts.snap_bytes});
    try w(&out, a, "block_tokens={d}", .{block_tokens});
    try w(&out, a, "planes={s}", .{parts.planes});
    return out.toOwnedSlice(a);
}

/// The chain's seed text: only what the handshake makes equal on both ranks (never the rank): both key prompts alike.
pub fn seedText(a: std.mem.Allocator, parts: Parts) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const w = struct {
        fn f(o: *std.ArrayList(u8), al: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
            const t = try std.fmt.allocPrint(al, fmt ++ "\n", args);
            defer al.free(t);
            try o.appendSlice(al, t);
        }
    }.f;
    try w(&out, a, "chain format={d}", .{pd.version});
    try w(&out, a, "world={d}", .{parts.world});
    try w(&out, a, "deploy={s}", .{parts.deploy});
    try w(&out, a, "bundle={s}", .{parts.bundle});
    try w(&out, a, "kit_sha256={s}", .{parts.kit_sha256});
    var it = std.mem.tokenizeScalar(u8, parts.plan, ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, "conf=") or std.mem.startsWith(u8, tok, "disk=")) continue;
        try w(&out, a, "plan.{s}", .{tok});
    }
    try w(&out, a, "block_tokens={d}", .{block_tokens});
    return out.toOwnedSlice(a);
}

pub const Parts = struct {
    rank: u32,
    world: u32,
    deploy: []const u8,
    bundle: []const u8,
    kit_sha256: []const u8,
    hip_runtime: c_int,
    arch: []const u8,
    plan: []const u8,
    engram: []const u8,
    token_map: []const u8,
    ds_kit: []const u8,
    window: usize,
    snap_bytes: usize,
    planes: []const u8,
};

/// A served start's launch identity (serve_lib.sh): without the deployed commit and bundle hash no state is trusted.
fn launchValue(name: [:0]const u8) ?[]const u8 {
    const v = envVar(name) orelse return null;
    return if (v.len == 0 or std.mem.eql(u8, v, "-")) null else v;
}

/// The SHA-256 of a file's bytes (hex), or "missing".
fn fileSha(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30)) catch return "missing";
    defer a.free(bytes);
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return a.dupe(u8, &std.fmt.bytesToHex(d, .lower));
}

/// A directory's regular files as "name:size,..." sorted by name (the Engram tables: their names and sizes).
fn listing(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![]const u8 {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return "missing";
    defer d.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var it = d.iterate();
    while (try it.next(io)) |e| if (e.kind == .file) try names.append(a, try a.dupe(u8, e.name));
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    var out: std.ArrayList(u8) = .empty;
    for (names.items, 0..) |n, i| {
        const st = d.statFile(io, n, .{}) catch continue;
        const s = try std.fmt.allocPrint(a, "{s}{s}:{d}", .{ if (i == 0) "" else ",", n, st.size });
        try out.appendSlice(a, s);
    }
    return out.toOwnedSlice(a);
}

/// This rank's identity parts (arena `a`): identityText names its directory, seedText seeds the prompt chain.
pub fn identity(a: std.mem.Allocator, m: *const Model, planes: []const Plane) !Parts {
    const deploy = launchValue("TF_DS_DEPLOY") orelse return error.NoLaunchIdentity;
    const bundle = launchValue("TF_DS_BUNDLE") orelse return error.NoLaunchIdentity;
    var arch_buf: [256]u8 = undefined;
    // copied into the arena: Parts outlives this frame, and a slice of arch_buf would change every start
    const arch = try a.dupe(u8, m.ctx.arch(&arch_buf) catch "unknown");
    var pl: std.ArrayList(u8) = .empty;
    for (planes, 0..) |p, i| try pl.appendSlice(a, try std.fmt.allocPrint(a, "{s}{d}:{d}", .{ if (i == 0) "" else ",", p.row, p.ratio }));
    var ds_kit: std.ArrayList(u8) = .empty;
    try ds_kit.appendSlice(a, try std.fmt.allocPrint(a, "engram.json:{s}", .{try fileSha(a, m.io, try std.fs.path.join(a, &.{ m.kit_dir, "engram.json" }))}));
    // the RoPE tables by size (the deploy's own files, under its commit; hashing them would read up to 512 MiB a start)
    for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }) |n| {
        const st = std.Io.Dir.cwd().statFile(m.io, try std.fs.path.join(a, &.{ m.kit_dir, n }), .{}) catch {
            try ds_kit.appendSlice(a, try std.fmt.allocPrint(a, ",{s}:missing", .{n}));
            continue;
        };
        try ds_kit.appendSlice(a, try std.fmt.allocPrint(a, ",{s}:{d}", .{ n, st.size }));
    }
    const engram = if (m.engram_dir) |e| try std.fmt.allocPrint(a, "{s}[{s}]", .{ std.fs.path.basename(e), try listing(a, m.io, e) }) else "none";
    const kit_sha: []const u8 = if (gpu.backend == .hip) &m.kit.manifest_sha256 else "cuda";
    return .{
        .rank = m.rank,
        .world = m.world,
        .deploy = deploy,
        .bundle = bundle,
        .kit_sha256 = kit_sha,
        .hip_runtime = m.ctx.d.version() catch 0,
        .arch = arch,
        .plan = m.plan_text,
        .engram = engram,
        .token_map = if (m.token_map) |t| try fileSha(a, m.io, t) else "none",
        .ds_kit = ds_kit.items,
        .window = m.cfg.window,
        .snap_bytes = m.snapBytes(),
        .planes = pl.items,
    };
}
