//! The AMD kit manifest (tools/dsv41-hip/ADAPTER.md section 7, schema tensorfold.dsv41.hip-kit/1): which code objects a
//! kit holds (by SHA-256), every kernel in them with its id, specialization, argument layout and launch facts read from
//! the object's own metadata, and the kernels each feature requires. Written by tf-hip-kit, read by the loader.

const std = @import("std");

pub const schema = "tensorfold.dsv41.hip-kit/1";
pub const registry_schema = "tensorfold.dsv41.hip-registry/1";

pub const Arg = struct { offset: u32, size: u32, kind: []const u8 };

pub const TritonParam = struct { name: []const u8, type: []const u8, div16: bool = false, nospec: bool = false };
pub const TritonConst = struct { int: ?i64 = null, f32: ?u32 = null };

/// A Triton variant's own facts (the fields aot.json carries today), for kernels with abi "triton".
pub const Triton = struct {
    @"fn": []const u8,
    params: []const TritonParam,
    consts: std.json.ArrayHashMap(TritonConst) = .{},
    num_warps: u32,
    shared: u32 = 0,
    /// Parameters Triton's launcher appends after the runtime ones (passed as null; their scratch sizes must be 0).
    hidden: []const []const u8 = &.{},
};

pub const Kernel = struct {
    id: []const u8,
    symbol: []const u8,
    object: []const u8,
    /// "params": one C parameter per field of the Zig declaration; "triton": a Triton variant (see `triton`).
    abi: []const u8 = "params",
    spec: std.json.ArrayHashMap(i64) = .{},
    args: []const Arg,
    kernarg_size: u32,
    wave: u32,
    max_flat_workgroup_size: u32,
    lds_static: u32,
    scratch: u32,
    vgpr: u32,
    sgpr: u32,
    /// The launch block the kernel is written for (null: any up to max_flat_workgroup_size).
    block: ?[3]u32 = null,
    lds_dynamic_max: u32 = 0,
    owner: []const u8 = "",
    numerics: []const u8 = "",
    allow_scratch: bool = false,
    triton: ?Triton = null,
};

pub const Object = struct { file: []const u8, sha256: []const u8, bytes: u64, origin: []const u8, owner: []const u8 = "", e_flags: u32 };

pub const Toolchain = struct { hipcc: []const u8 = "", triton: []const u8 = "", flags: []const []const u8 = &.{} };
/// A file the kit was built from (each .hip source, abi.h, the registry), by content: the kit names exactly what made
/// it, independent of the commit it sits in (a run's receipt records the git state).
pub const Input = struct { name: []const u8, sha256: []const u8 };

/// Kernel ids each feature needs; a loader resolves `always` plus the features its configuration turns on.
pub const Required = struct {
    always: []const []const u8 = &.{},
    drafts: []const []const u8 = &.{},
    engram: []const []const u8 = &.{},
    profile: []const []const u8 = &.{},
    graphs: []const []const u8 = &.{},
};

pub const Manifest = struct {
    schema: []const u8,
    kit: []const u8,
    abi_version: u32,
    arch: []const u8,
    wave: u32,
    inputs: []const Input = &.{},
    toolchain: Toolchain = .{},
    objects: []const Object,
    kernels: []const Kernel,
    required: Required = .{},

    pub fn parse(a: std.mem.Allocator, text: []const u8) !Manifest {
        const m = try std.json.parseFromSliceLeaky(Manifest, a, text, .{ .allocate = .alloc_always });
        if (!std.mem.eql(u8, m.schema, schema)) return error.ManifestSchema;
        return m;
    }

    pub fn object(m: Manifest, file: []const u8) ?Object {
        for (m.objects) |o| if (std.mem.eql(u8, o.file, file)) return o;
        return null;
    }
};

/// The registry a kit is built from (zig/kernels/hip/dsv41/registry.json): per source file, the kernels it must define.
pub const RegistryKernel = struct {
    id: []const u8,
    symbol: []const u8,
    spec: std.json.ArrayHashMap(i64) = .{},
    block: ?[3]u32 = null,
    lds_dynamic_max: u32 = 0,
    numerics: []const u8 = "",
    allow_scratch: bool = false,
    /// The kernel may compile an fp16 conversion of an fp32 product into a mixed-precision fma (v_fma_mixlo_f16 /
    /// v_fma_mixhi_f16: one rounding where the source has two); tf-hip-kit isa-lint refuses it otherwise.
    allow_fma_mix: bool = false,
    feature: []const u8 = "always",
    /// A Triton variant's metadata (w-tri's AOT step writes these); the manifest entry then has abi "triton".
    triton: ?Triton = null,
};

pub const RegistrySource = struct { source: []const u8, owner: []const u8, kit: []const u8, kernels: []const RegistryKernel };
/// `required`: kernel ids a kit must hold by feature, declared apart from what any source provides (so a kit missing a
/// kernel the configuration reaches fails to open instead of passing because nobody listed it).
pub const Registry = struct { schema: []const u8, sources: []const RegistrySource, required: std.json.ArrayHashMap(Required) = .{} };

/// One variant of w-tri's AMD Triton kit (DIR/aot.json: the CUDA kit's KernelJson fields plus arch, warp_size, block,
/// file; extra fields such as isa_stats are ignored).
pub const TritonKitKernel = struct {
    @"fn": []const u8,
    hash: []const u8,
    name: []const u8,
    num_warps: u32,
    num_ctas: u32 = 1,
    shared: u32 = 0,
    global_scratch: u32 = 0,
    profile_scratch: u32 = 0,
    pdl: bool = false,
    params: []const TritonParam,
    consts: std.json.ArrayHashMap(TritonConst) = .{},
    arch: []const u8,
    warp_size: u32,
    block: u32,
    file: []const u8,
};
pub const TritonKit = struct { arch: []const u8, kernels: []const TritonKitKernel };

/// Bytes a Triton parameter of `ty` takes in the kernarg segment (pointers 8, i64/u64/fp64 8, i32/fp32 4).
pub fn tritonParamSize(ty: []const u8) u32 {
    if (ty.len > 0 and ty[0] == '*') return 8;
    if (std.mem.eql(u8, ty, "u64") or std.mem.eql(u8, ty, "i64") or std.mem.eql(u8, ty, "fp64")) return 8;
    return 4;
}

/// The two pointers Triton 3.8's AMD launcher appends after the runtime parameters.
pub const triton_hidden = [_][]const u8{ "global_scratch", "profile_scratch" };

/// Lower-case hex SHA-256 of `bytes`.
pub fn sha256Hex(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return std.fmt.bytesToHex(h, .lower);
}

/// Whether a manifest spec equals a requested one exactly (same keys, same values).
pub fn sameSpec(have: std.json.ArrayHashMap(i64), want: []const SpecItem) bool {
    if (have.map.count() != want.len) return false;
    for (want) |w| {
        const v = have.map.get(w.key) orelse return false;
        if (v != w.value) return false;
    }
    return true;
}

pub const SpecItem = struct { key: []const u8, value: i64 };

test "a manifest round-trips through JSON and refuses another schema" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var spec: std.json.ArrayHashMap(i64) = .{};
    try spec.map.put(a, "k2", 12);
    const m: Manifest = .{
        .schema = schema,
        .kit = "test",
        .abi_version = 1,
        .arch = "gfx1151",
        .wave = 32,
        .objects = &.{.{ .file = "objects/ab.hsaco", .sha256 = "ab", .bytes = 10, .origin = "x.hip", .e_flags = 0x4a }},
        .kernels = &.{.{ .id = "x.y", .symbol = "tf_x", .object = "objects/ab.hsaco", .spec = spec, .args = &.{.{ .offset = 0, .size = 8, .kind = "global_buffer" }}, .kernarg_size = 8, .wave = 32, .max_flat_workgroup_size = 256, .lds_static = 0, .scratch = 0, .vgpr = 4, .sgpr = 8 }},
        .required = .{ .always = &.{"x.y"} },
    };
    const text = try std.json.Stringify.valueAlloc(a, m, .{ .whitespace = .indent_2 });
    const back = try Manifest.parse(a, text);
    try std.testing.expectEqualStrings("tf_x", back.kernels[0].symbol);
    try std.testing.expect(sameSpec(back.kernels[0].spec, &.{.{ .key = "k2", .value = 12 }}));
    try std.testing.expect(!sameSpec(back.kernels[0].spec, &.{.{ .key = "k2", .value = 10 }}));
    try std.testing.expect(!sameSpec(back.kernels[0].spec, &.{}));
    try std.testing.expectEqual(@as(u32, 0x4a), back.object("objects/ab.hsaco").?.e_flags);
    const other = try std.mem.replaceOwned(u8, a, text, schema, "tensorfold.other/9");
    try std.testing.expectError(error.ManifestSchema, Manifest.parse(a, other));
}

test "sha256 is the lower-case hex digest" {
    try std.testing.expectEqualStrings("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", &sha256Hex("abc"));
}
