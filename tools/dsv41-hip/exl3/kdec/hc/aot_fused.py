"""aot_fused.py SRC_KIT OUT_KIT: OUT_KIT = SRC_KIT (gfx1151-v5: aot.json + hsaco/) plus hc_fuse._hc_fused AOT-compiled for
gfx1151 (wave32) for every decode / verify window of 1-16 rows: one variant a (RB, rows' specialization) as the engine's
tri_hc.fused launches it (rows 1 folded to a constexpr as Triton does; rows % 16 == 0 specialized), entries in
aot_build.py's schema (tools/dsv41-hip/tri; the AMD fields: arch, warp_size, block, file, sha256)."""
import hashlib, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.home() / "tfp/tri/src"))
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import hc_fuse as F

src, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    sys.exit(f"{out} exists")
shutil.copytree(src, out)
m = json.load(open(out / "aot.json"))
P = [("X", "*bf16"), ("FN", "*fp32"), ("PART", "*fp32"), ("CNT", "*i32"), ("BASE", "*fp32"), ("SCALE", "*fp32"),
     ("PRE_IN", "*fp32"), ("NW", "*bf16"), ("OUT", "*bf16"), ("PRE_OUT", "*fp32"), ("POST", "*fp32"), ("COMB", "*fp32"),
     ("eps", "fp32"), ("hc_eps", "fp32")]
base_consts = {"WIDE": 20480, "D": 5120, "NB": 40, "SUB": 128, "MB": 8, "ITERS": 20, "BLOCK": 1024, "SST": 1}
added = []
for rb, rows_kind in ((1, "one"), (2, "i32"), (3, "i32"), (4, "i32"), (8, "i32"), (8, "i32d16")):
    fn = F._hc_fused
    sig, constexprs, attrs, params = {}, {}, {}, []
    consts = dict(base_consts, RB=rb)
    for i, name in enumerate(fn.arg_names):
        if name in consts:
            sig[name] = "constexpr"
            constexprs[name] = consts[name]
        elif name == "rows":
            if rows_kind == "one":
                sig[name] = "constexpr"
                constexprs[name] = 1
                consts["rows"] = 1
            else:
                sig[name] = "i32"
                d16 = rows_kind == "i32d16"
                if d16:
                    attrs[(i,)] = [["tt.divisibility", 16]]
                params.append({"name": name, "type": "i32", "div16": d16, "nospec": False})
        else:
            ty = dict(P)[name]
            sig[name] = ty
            if ty.startswith("*"):
                attrs[(i,)] = [["tt.divisibility", 16]]
            params.append({"name": name, "type": ty, "div16": ty.startswith("*"), "nospec": False})
    k = triton.compile(ASTSource(fn, sig, constexprs=constexprs, attrs=attrs), target=GPUTarget("hip", "gfx1151", 32),
                       options={"num_warps": 8})
    hsaco = k.asm["hsaco"]
    sha = hashlib.sha256(hsaco).hexdigest()
    (out / "hsaco" / f"{sha}.hsaco").write_bytes(hsaco)
    md = k.metadata
    stats = {}
    for line in k.asm.get("amdgcn", "").splitlines():
        line = line.strip()
        for key in (".vgpr_count:", ".sgpr_count:", ".vgpr_spill_count:", ".sgpr_spill_count:",
                    ".private_segment_fixed_size:", ".group_segment_fixed_size:"):
            if line.startswith(key):
                stats[key.strip(".:")] = int(line.split(":")[1].strip().split()[0])
    e = {"fn": "_hc_fused", "hash": sha, "name": md.name, "num_warps": md.num_warps, "num_ctas": md.num_ctas,
         "shared": md.shared, "global_scratch": getattr(md, "global_scratch_size", 0), "global_align": getattr(md, "global_scratch_align", 1),
         "profile_scratch": getattr(md, "profile_scratch_size", 0), "pdl": False, "params": params,
         "consts": {kk: {"int": vv} for kk, vv in consts.items()}, "arch": "gfx1151", "warp_size": md.warp_size,
         "block": md.num_warps * md.warp_size, "file": f"hsaco/{sha}.hsaco", "options": {"num_warps": 8},
         "isa_stats": stats, "recorded_grid": [1, 41, 3], "origin": "k-dec2 hc_fuse.py (tools/dsv41-hip/exl3/kdec)"}
    m["kernels"].append(e)
    added.append((rb, rows_kind, sha[:12], stats))
    print("_hc_fused RB", rb, rows_kind, sha[:12], stats, "shared", md.shared, flush=True)
m["kdec2_added"] = {"fn": "_hc_fused", "source_sha256": hashlib.sha256(open(F.__file__, "rb").read()).hexdigest()}
json.dump(m, open(out / "aot.json", "w"), indent=1)
print("kit", out, "kernels", len(m["kernels"]))
