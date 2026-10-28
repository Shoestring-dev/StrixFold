//! tf-hip-kit: builds and checks AMD kits on the CPU (ADAPTER.md 7.3). No GPU, no HIP runtime.
//!   tf-hip-kit scan --registry FILE --kit NAME --arch gfx1151 --out DIR [--hipcc-version-file F] [--flag F]...
//!                   [--input NAME=PATH]... [--triton DIR] [--required registry|declared] SOURCE=OBJECT...
//!       --required declared: the manifest requires only the kernels its sources declare (a partial kit for tests);
//!       registry (default) adds the registry's `required` lists for the kit, which a complete kit must hold
//!       --triton: w-tri's AOT Triton kit (DIR/aot.json, DIR/hsaco/<sha256>.hsaco), each variant checked against its
//!       object's metadata and added with abi "triton" (spills allowed and counted)
//!       every kernel the registry declares for SOURCE must be in OBJECT and every kernel in OBJECT must be declared;
//!       writes DIR/objects/<sha256>.hsaco and DIR/manifest.json with each kernel's metadata from the object itself
//!   tf-hip-kit verify DIR [--arch gfx1151]
//!       re-hashes and re-parses every object and checks the manifest against it; exit 1 on any difference
//!   tf-hip-kit show DIR
//!       one line a kernel: id, spec, symbol, explicit args, block bound, LDS, scratch, registers
//!   tf-hip-kit lint DIR...
//!       device sources (.hip .h .hpp .cuh) must not use __builtin_bit_cast / std::bit_cast unless the line says
//!       "bitcast-ok: <reason>": clang 23 / HIP 7.15 on gfx1151 returns lane 0 for __builtin_bit_cast(uint16_t,
//!       f16x2[1]) at every -O level (found by w-exl3), so every use is reviewed by hand
//!   tf-hip-kit isa-lint --registry FILE SOURCE=DISASSEMBLY...
//!       the compiled native kernels (llvm-objdump -d of each source's code object) must not contain v_fma_mixlo_f16 /
//!       v_fma_mixhi_f16 (v_mad_mix*_f16 on older targets) unless the kernel's registry entry sets allow_fma_mix:
//!       clang 23 / HIP 7.15 compiles __float2half_rn(a * b) of fp32 a, b into one, a single rounding of the exact
//!       product where the source rounds to fp32 and then to fp16 (found by w-exl3: 1-ulp differences on fp16 ties);
//!       an fp32 value made opaque first (asm volatile("" : "+v"(x))) keeps both roundings

const std = @import("std");
const manifest = @import("manifest.zig");
const hsaco = @import("hsaco.zig");
const abi_h = @import("dsv41_abi");

const usage =
    \\usage: tf-hip-kit scan --registry FILE --kit NAME --arch ARCH --out DIR [--hipcc-version-file F] [--flag F]... [--input NAME=PATH]... [--triton DIR] [--required registry|declared] SOURCE=OBJECT...
    \\       tf-hip-kit verify DIR [--arch ARCH]
    \\       tf-hip-kit show DIR
    \\       tf-hip-kit lint DIR...
    \\       tf-hip-kit isa-lint --registry FILE SOURCE=DISASSEMBLY...
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 3) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    if (std.mem.eql(u8, args[1], "scan")) return scan(a, io, args[2..]);
    if (std.mem.eql(u8, args[1], "verify")) return verify(a, io, args[2], flag(args[3..], "--arch") orelse "gfx1151", true);
    if (std.mem.eql(u8, args[1], "show")) return verify(a, io, args[2], null, false);
    if (std.mem.eql(u8, args[1], "lint")) return lint(a, io, args[2..]);
    if (std.mem.eql(u8, args[1], "isa-lint")) return isaLint(a, io, args[2..]);
    std.debug.print("{s}", .{usage});
    return 2;
}

fn flag(args: []const [:0]const u8, name: []const u8) ?[]const u8 {
    for (args, 0..) |x, i| if (std.mem.eql(u8, x, name) and i + 1 < args.len) return args[i + 1];
    return null;
}

fn scan(a: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !u8 {
    const registry_path = flag(args, "--registry") orelse return fail("scan needs --registry");
    const kit_name = flag(args, "--kit") orelse return fail("scan needs --kit");
    const arch = flag(args, "--arch") orelse return fail("scan needs --arch");
    const out = flag(args, "--out") orelse return fail("scan needs --out");
    const want_mach = hsaco.machOf(arch) orelse return fail("unknown --arch");
    const required_mode = flag(args, "--required") orelse "registry";
    const declared_only = std.mem.eql(u8, required_mode, "declared");
    if (!declared_only and !std.mem.eql(u8, required_mode, "registry")) return fail("--required registry or declared");
    var flags: std.ArrayList([]const u8) = .empty;
    var pairs: std.ArrayList([2][]const u8) = .empty;
    var inputs: std.ArrayList(manifest.Input) = .empty;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const x = args[i];
        if (std.mem.startsWith(u8, x, "--")) {
            if (std.mem.eql(u8, x, "--flag") and i + 1 < args.len) try flags.append(a, args[i + 1]);
            if (std.mem.eql(u8, x, "--input") and i + 1 < args.len) {
                const spec = args[i + 1];
                const eq = std.mem.indexOfScalar(u8, spec, '=') orelse return fail("expected --input NAME=PATH");
                const text = try std.Io.Dir.cwd().readFileAlloc(io, spec[eq + 1 ..], a, .limited(64 << 20));
                try inputs.append(a, .{ .name = spec[0..eq], .sha256 = try a.dupe(u8, &manifest.sha256Hex(text)) });
            }
            i += 1;
            continue;
        }
        const eq = std.mem.indexOfScalar(u8, x, '=') orelse return fail("expected SOURCE=OBJECT");
        try pairs.append(a, .{ x[0..eq], x[eq + 1 ..] });
    }
    const reg = std.json.parseFromSliceLeaky(manifest.Registry, a, try std.Io.Dir.cwd().readFileAlloc(io, registry_path, a, .limited(16 << 20)), .{ .allocate = .alloc_always }) catch |e| {
        std.debug.print("{s}: not a registry ({s})\n", .{ registry_path, @errorName(e) });
        return 1;
    };
    if (!std.mem.eql(u8, reg.schema, manifest.registry_schema)) return fail("registry schema");
    try std.Io.Dir.cwd().createDirPath(io, try std.fs.path.join(a, &.{ out, "objects" }));
    var objects: std.ArrayList(manifest.Object) = .empty;
    var kernels: std.ArrayList(manifest.Kernel) = .empty;
    var req: [5]std.ArrayList([]const u8) = @splat(.empty);
    const feature_names = [_][]const u8{ "always", "drafts", "engram", "profile", "graphs" };
    var bad: usize = 0;
    for (pairs.items) |pair| {
        const src = for (reg.sources) |s| {
            if (std.mem.eql(u8, s.source, pair[0]) and std.mem.eql(u8, s.kit, kit_name)) break s;
        } else {
            std.debug.print("{s}: not a source of kit {s} in the registry\n", .{ pair[0], kit_name });
            return 1;
        };
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, pair[1], a, .limited(1 << 30));
        const co = try hsaco.parse(a, bytes);
        if (co.mach() != want_mach) {
            std.debug.print("{s}: e_flags {x} is not {s}\n", .{ pair[1], co.e_flags, arch });
            return 1;
        }
        const sum = manifest.sha256Hex(bytes);
        const file = try std.fmt.allocPrint(a, "objects/{s}.hsaco", .{sum});
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ out, file }), .data = bytes });
        try objects.append(a, .{ .file = file, .sha256 = try a.dupe(u8, &sum), .bytes = bytes.len, .origin = src.source, .owner = src.owner, .e_flags = co.e_flags });
        // every compiled kernel is declared, every declared kernel is compiled
        for (co.kernels) |k| {
            const declared = for (src.kernels) |rk| {
                if (std.mem.eql(u8, rk.symbol, k.name)) break true;
            } else false;
            if (!declared) {
                std.debug.print("{s}: kernel {s} is not declared in the registry\n", .{ src.source, k.name });
                bad += 1;
            }
        }
        for (src.kernels) |rk| {
            const k = co.find(rk.symbol) orelse {
                std.debug.print("{s}: {s} ({s}) is declared but not compiled\n", .{ src.source, rk.symbol, rk.id });
                bad += 1;
                continue;
            };
            if (k.wave != 32 or k.dynamic_stack or (k.scratch > 0 and !rk.allow_scratch)) {
                std.debug.print("{s}: {s} wave {d}, dynamic stack {}, scratch {d} B (spills; declare allow_scratch to accept)\n", .{ src.source, rk.symbol, k.wave, k.dynamic_stack, k.scratch });
                bad += 1;
            }
            if (rk.block) |b| if (b[0] * b[1] * b[2] > k.max_flat_workgroup_size) {
                std.debug.print("{s}: {s} declares a {d}-thread block above its launch bound {d}\n", .{ src.source, rk.symbol, b[0] * b[1] * b[2], k.max_flat_workgroup_size });
                bad += 1;
            };
            const margs = try a.alloc(manifest.Arg, k.args.len);
            for (k.args, margs) |x, *y| y.* = .{ .offset = x.offset, .size = x.size, .kind = x.kind };
            try kernels.append(a, .{
                .id = rk.id,
                .symbol = rk.symbol,
                .object = file,
                .spec = rk.spec,
                .args = margs,
                .kernarg_size = k.kernarg_size,
                .wave = k.wave,
                .max_flat_workgroup_size = k.max_flat_workgroup_size,
                .lds_static = k.lds_static,
                .scratch = k.scratch,
                .vgpr = k.vgpr,
                .sgpr = k.sgpr,
                .block = rk.block,
                // a Triton variant's metadata names the dynamic LDS every launch of it takes
                .lds_dynamic_max = if (rk.triton) |t| @max(rk.lds_dynamic_max, t.shared) else rk.lds_dynamic_max,
                .owner = src.owner,
                .numerics = rk.numerics,
                .allow_scratch = rk.allow_scratch,
                .abi = if (rk.triton != null) "triton" else "params",
                .triton = rk.triton,
            });
            for (feature_names, &req) |fname, *list| if (std.mem.eql(u8, fname, rk.feature)) {
                for (list.items) |id| {
                    if (std.mem.eql(u8, id, rk.id)) break;
                } else try list.append(a, rk.id);
            };
        }
    }
    if (flag(args, "--triton")) |tdir| bad += try importTriton(a, io, tdir, arch, want_mach, out, &objects, &kernels, &inputs);
    if (!declared_only) if (reg.required.map.get(kit_name)) |r| {
        const declared = [_][]const []const u8{ r.always, r.drafts, r.engram, r.profile, r.graphs };
        for (declared, &req) |ids, *list| for (ids) |id| {
            for (list.items) |have| {
                if (std.mem.eql(u8, have, id)) break;
            } else try list.append(a, id);
        };
    };
    if (bad > 0) return 1;
    const m: manifest.Manifest = .{
        .schema = manifest.schema,
        .kit = kit_name,
        .abi_version = abi_h.TF_DS_HIP_ABI_VERSION,
        .arch = arch,
        .wave = 32,
        .inputs = inputs.items,
        .toolchain = .{ .hipcc = try versionLine(a, io, flag(args, "--hipcc-version-file")), .flags = flags.items },
        .objects = objects.items,
        .kernels = kernels.items,
        .required = .{ .always = req[0].items, .drafts = req[1].items, .engram = req[2].items, .profile = req[3].items, .graphs = req[4].items },
    };
    const text = try std.json.Stringify.valueAlloc(a, m, .{ .whitespace = .indent_1 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ out, "manifest.json" }), .data = text });
    std.debug.print("kit {s}: {d} objects, {d} kernels, manifest sha256 {s}\n", .{ kit_name, objects.items.len, kernels.items.len, &manifest.sha256Hex(text) });
    return 0;
}

/// Adds every variant of an AMD Triton kit; returns the number of refused variants (each named on stderr).
fn importTriton(a: std.mem.Allocator, io: std.Io, dir: []const u8, arch: []const u8, want_mach: u32, out: []const u8, objects: *std.ArrayList(manifest.Object), kernels: *std.ArrayList(manifest.Kernel), inputs: *std.ArrayList(manifest.Input)) !usize {
    const cwd = std.Io.Dir.cwd();
    const text = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "aot.json" }), a, .limited(64 << 20));
    try inputs.append(a, .{ .name = "triton/aot.json", .sha256 = try a.dupe(u8, &manifest.sha256Hex(text)) });
    const tk = try std.json.parseFromSliceLeaky(manifest.TritonKit, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    var bad: usize = 0;
    var spills: usize = 0;
    if (!std.mem.eql(u8, tk.arch, arch)) {
        std.debug.print("{s}: Triton kit for {s}, not {s}\n", .{ dir, tk.arch, arch });
        return 1;
    }
    for (tk.kernels) |v| {
        if (!std.mem.eql(u8, v.arch, arch) or v.warp_size != 32 or v.block != v.num_warps * 32 or v.num_ctas != 1 or v.pdl or v.global_scratch != 0 or v.profile_scratch != 0) {
            std.debug.print("{s} ({s}): arch {s}, wave {d}, block {d} for {d} warps, {d} CTAs, pdl {}, scratch {d}/{d}: not loadable here\n", .{ v.@"fn", v.hash, v.arch, v.warp_size, v.block, v.num_warps, v.num_ctas, v.pdl, v.global_scratch, v.profile_scratch });
            bad += 1;
            continue;
        }
        const bytes = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, v.file }), a, .limited(1 << 30));
        const sum = manifest.sha256Hex(bytes);
        const co = try hsaco.parse(a, bytes);
        if (co.mach() != want_mach) {
            std.debug.print("{s}: e_flags {x} is not {s}\n", .{ v.file, co.e_flags, arch });
            bad += 1;
            continue;
        }
        const k = co.find(v.name) orelse {
            std.debug.print("{s}: kernel {s} not in {s}\n", .{ v.@"fn", v.name, v.file });
            bad += 1;
            continue;
        };
        const ex = k.explicit();
        var ok = ex.len == v.params.len + manifest.triton_hidden.len and k.wave == 32 and !k.dynamic_stack;
        if (ok) for (v.params, ex[0..v.params.len]) |p, x| {
            ok = ok and x.size == manifest.tritonParamSize(p.type);
        };
        if (ok) for (ex[v.params.len..]) |x| {
            ok = ok and x.size == 8;
        };
        if (!ok) {
            std.debug.print("{s} ({s}): {d} explicit args for {d} params + {d} scratch pointers, or their sizes, wave {d}, dynamic stack {}: refused\n", .{ v.@"fn", v.hash, ex.len, v.params.len, manifest.triton_hidden.len, k.wave, k.dynamic_stack });
            bad += 1;
            continue;
        }
        if (k.scratch > 0) spills += 1;
        const file = try std.fmt.allocPrint(a, "objects/{s}.hsaco", .{sum});
        const have = for (objects.items) |o| {
            if (std.mem.eql(u8, o.file, file)) break true;
        } else false;
        if (!have) {
            try cwd.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ out, file }), .data = bytes });
            try objects.append(a, .{ .file = file, .sha256 = try a.dupe(u8, &sum), .bytes = bytes.len, .origin = try std.fmt.allocPrint(a, "triton:{s}", .{v.file}), .owner = "w-tri", .e_flags = co.e_flags });
        }
        const margs = try a.alloc(manifest.Arg, k.args.len);
        for (k.args, margs) |x, *y| y.* = .{ .offset = x.offset, .size = x.size, .kind = x.kind };
        try kernels.append(a, .{
            .id = try std.fmt.allocPrint(a, "tri.{s}", .{v.@"fn"}),
            .symbol = v.name,
            .object = file,
            .abi = "triton",
            .args = margs,
            .kernarg_size = k.kernarg_size,
            .wave = k.wave,
            .max_flat_workgroup_size = k.max_flat_workgroup_size,
            .lds_static = k.lds_static,
            .scratch = k.scratch,
            .vgpr = k.vgpr,
            .sgpr = k.sgpr,
            .lds_dynamic_max = v.shared,
            .owner = "w-tri",
            .allow_scratch = true, // spills cost speed, not correctness: counted here, w-tri's to tune
            .triton = .{ .@"fn" = v.@"fn", .params = v.params, .consts = v.consts, .num_warps = v.num_warps, .shared = v.shared, .hidden = &manifest.triton_hidden },
        });
    }
    std.debug.print("triton {s}: {d} variants, {d} refused, {d} spill registers to scratch\n", .{ dir, tk.kernels.len, bad, spills });
    return bad;
}

/// verify (arch given) or show: every object hashed and parsed again, every manifest entry compared with it.
fn verify(a: std.mem.Allocator, io: std.Io, dir: []const u8, arch: ?[]const u8, check: bool) !u8 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "manifest.json" }), a, .limited(64 << 20));
    const m = manifest.Manifest.parse(a, text) catch |e| {
        std.debug.print("{s}: not a kit manifest ({s})\n", .{ dir, @errorName(e) });
        return 1;
    };
    var bad: usize = 0;
    if (check) {
        if (m.abi_version != abi_h.TF_DS_HIP_ABI_VERSION) {
            std.debug.print("ABI version {d}, this tool's {d}\n", .{ m.abi_version, abi_h.TF_DS_HIP_ABI_VERSION });
            bad += 1;
        }
        if (!std.mem.eql(u8, m.arch, arch.?)) {
            std.debug.print("arch {s}, wanted {s}\n", .{ m.arch, arch.? });
            bad += 1;
        }
    }
    const cos = try a.alloc(?hsaco.CodeObject, m.objects.len);
    for (m.objects, cos) |o, *co| {
        co.* = null;
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, o.file }), a, .limited(1 << 30)) catch {
            std.debug.print("{s}: missing\n", .{o.file});
            bad += 1;
            continue;
        };
        if (!std.mem.eql(u8, &manifest.sha256Hex(bytes), o.sha256) or bytes.len != o.bytes) {
            std.debug.print("{s}: SHA-256 or size differs from the manifest\n", .{o.file});
            bad += 1;
            continue;
        }
        co.* = hsaco.parse(a, bytes) catch |e| {
            std.debug.print("{s}: {s}\n", .{ o.file, @errorName(e) });
            bad += 1;
            continue;
        };
        if (co.*.?.e_flags != o.e_flags or (check and co.*.?.mach() != (hsaco.machOf(arch.?) orelse 0))) {
            std.debug.print("{s}: e_flags {x}\n", .{ o.file, co.*.?.e_flags });
            bad += 1;
        }
    }
    for (m.kernels) |k| {
        const oi = for (m.objects, 0..) |o, j| {
            if (std.mem.eql(u8, o.file, k.object)) break j;
        } else {
            std.debug.print("{s}: object {s} not listed\n", .{ k.id, k.object });
            bad += 1;
            continue;
        };
        const co = cos[oi] orelse continue;
        const meta = co.find(k.symbol) orelse {
            std.debug.print("{s}: {s} not in {s}\n", .{ k.id, k.symbol, k.object });
            bad += 1;
            continue;
        };
        var same = meta.args.len == k.args.len and meta.kernarg_size == k.kernarg_size and meta.wave == k.wave and meta.max_flat_workgroup_size == k.max_flat_workgroup_size;
        if (same) for (meta.args, k.args) |x, y| {
            same = same and x.offset == y.offset and x.size == y.size and std.mem.eql(u8, x.kind, y.kind);
        };
        if (!same) {
            std.debug.print("{s}: manifest disagrees with the object's metadata\n", .{k.id});
            bad += 1;
        }
        if (!check) {
            std.debug.print("{s} {s} spec", .{ k.id, k.symbol });
            var it = k.spec.map.iterator();
            while (it.next()) |e| std.debug.print(" {s}={d}", .{ e.key_ptr.*, e.value_ptr.* });
            std.debug.print(" args", .{});
            for (meta.explicit()) |x| std.debug.print(" {d}+{d}:{s}", .{ x.offset, x.size, x.kind });
            std.debug.print(" bound {d} lds {d}+{d} scratch {d} vgpr {d} sgpr {d}\n", .{ k.max_flat_workgroup_size, k.lds_static, k.lds_dynamic_max, k.scratch, k.vgpr, k.sgpr });
        }
    }
    const ids = [_][]const []const u8{ m.required.always, m.required.drafts, m.required.engram, m.required.profile, m.required.graphs };
    for (ids) |list| for (list) |id| {
        for (m.kernels) |k| {
            if (std.mem.eql(u8, k.id, id)) break;
        } else {
            std.debug.print("required {s} has no kernel\n", .{id});
            bad += 1;
        }
    };
    std.debug.print("kit {s} ({s}, ABI {d}): {d} objects, {d} kernels, manifest sha256 {s}: {s}\n", .{ m.kit, m.arch, m.abi_version, m.objects.len, m.kernels.len, &manifest.sha256Hex(text), if (bad == 0) "ok" else "FAILED" });
    return if (bad == 0) 0 else 1;
}

/// The first line of hipcc --version's captured output ("" without one).
fn versionLine(a: std.mem.Allocator, io: std.Io, path: ?[]const u8) ![]const u8 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path orelse return "", a, .limited(1 << 20));
    return std.mem.trim(u8, text[0 .. std.mem.indexOfScalar(u8, text, '\n') orelse text.len], " \r");
}

/// Every bit_cast in device sources needs a reviewed "bitcast-ok:" note on its line (see the usage text).
fn lint(a: std.mem.Allocator, io: std.Io, dirs: []const [:0]const u8) !u8 {
    var hits: usize = 0;
    var files: usize = 0;
    for (dirs) |root| {
        var dir = try std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
        defer dir.close(io);
        var walker = try dir.walk(a);
        defer walker.deinit();
        while (try walker.next(io)) |e| {
            if (e.kind != .file) continue;
            const ext = std.fs.path.extension(e.basename);
            const device = for ([_][]const u8{ ".hip", ".h", ".hpp", ".cuh" }) |x| {
                if (std.mem.eql(u8, ext, x)) break true;
            } else false;
            if (!device) continue;
            files += 1;
            const text = try e.dir.readFileAlloc(io, e.basename, a, .limited(16 << 20));
            var lines = std.mem.splitScalar(u8, text, '\n');
            var n: usize = 0;
            while (lines.next()) |line| {
                n += 1;
                const uses = std.mem.indexOf(u8, line, "__builtin_bit_cast") != null or std.mem.indexOf(u8, line, "bit_cast<") != null;
                if (!uses or std.mem.indexOf(u8, line, "bitcast-ok:") != null) continue;
                std.debug.print("{s}/{s}:{d}: bit_cast without a bitcast-ok note (clang 23 f16x2 lane bug): {s}\n", .{ root, e.path, n, std.mem.trim(u8, line, " ") });
                hits += 1;
            }
        }
    }
    std.debug.print("lint: {d} device source files, {d} unreviewed bit_cast uses\n", .{ files, hits });
    return if (hits == 0) 0 else 1;
}

/// The mnemonics isa-lint refuses: an fp16 result of a mixed-precision fma (see the usage text).
const fma_mix_f16 = [_][]const u8{ "v_fma_mixlo_f16", "v_fma_mixhi_f16", "v_mad_mixlo_f16", "v_mad_mixhi_f16" };

/// No native kernel folds an fp16 conversion into a mixed-precision fma unless its registry entry allows it (see the
/// usage text). Each argument after --registry is SOURCE=DISASSEMBLY: the registry's source name and llvm-objdump -d
/// of its code object, whose "<symbol>:" lines start each kernel.
fn isaLint(a: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !u8 {
    const registry_path = flag(args, "--registry") orelse return fail("isa-lint needs --registry");
    const reg = try std.json.parseFromSliceLeaky(manifest.Registry, a, try std.Io.Dir.cwd().readFileAlloc(io, registry_path, a, .limited(16 << 20)), .{ .allocate = .alloc_always });
    var hits: usize = 0;
    var functions: usize = 0;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--registry")) {
            i += 1;
            continue;
        }
        const eq = std.mem.indexOfScalar(u8, args[i], '=') orelse return fail("isa-lint takes SOURCE=DISASSEMBLY");
        const source = args[i][0..eq];
        const text = try std.Io.Dir.cwd().readFileAlloc(io, args[i][eq + 1 ..], a, .limited(512 << 20));
        var lines = std.mem.splitScalar(u8, text, '\n');
        var func: []const u8 = "";
        while (lines.next()) |raw| {
            const line = std.mem.trimEnd(u8, raw, " \r");
            if (std.mem.endsWith(u8, line, ">:")) {
                if (std.mem.indexOfScalar(u8, line, '<')) |lt| {
                    func = line[lt + 1 .. line.len - 2];
                    functions += 1;
                    continue;
                }
            }
            const insn = std.mem.trimStart(u8, line, " \t");
            const mixed = for (fma_mix_f16) |m| {
                if (std.mem.startsWith(u8, insn, m)) break true;
            } else false;
            if (!mixed or allowsFmaMix(reg, source, func)) continue;
            std.debug.print("{s}: {s}: {s} (an fp16 conversion folded into a mixed fma: one rounding where the source has two)\n", .{ source, func, insn });
            hits += 1;
        }
    }
    std.debug.print("isa-lint: {d} functions, {d} fp16 results of a mixed fma without allow_fma_mix\n", .{ functions, hits });
    return if (hits == 0) 0 else 1;
}

fn allowsFmaMix(reg: manifest.Registry, source: []const u8, symbol: []const u8) bool {
    for (reg.sources) |s| {
        if (!std.mem.eql(u8, s.source, source)) continue;
        for (s.kernels) |k| if (std.mem.eql(u8, k.symbol, symbol)) return k.allow_fma_mix;
    }
    return false;
}

test "isa-lint names an fp16 mixed fma unless the registry allows it" {
    const reg_text =
        \\{"schema": "tensorfold.dsv41.hip-registry/1", "sources": [{"source": "x.hip", "owner": "w", "kit": "k", "kernels": [
        \\  {"id": "x.a", "symbol": "tf_x_a"}, {"id": "x.b", "symbol": "tf_x_b", "allow_fma_mix": true}]}]}
    ;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const reg = try std.json.parseFromSliceLeaky(manifest.Registry, arena.allocator(), reg_text, .{ .allocate = .alloc_always });
    try std.testing.expect(!allowsFmaMix(reg, "x.hip", "tf_x_a"));
    try std.testing.expect(allowsFmaMix(reg, "x.hip", "tf_x_b"));
    try std.testing.expect(!allowsFmaMix(reg, "y.hip", "tf_x_b"));
}

fn fail(msg: []const u8) u8 {
    std.debug.print("tf-hip-kit: {s}\n{s}", .{ msg, usage });
    return 2;
}
