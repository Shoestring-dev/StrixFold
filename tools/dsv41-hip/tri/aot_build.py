"""AOT-compile every recorded Triton launch variant (TensorFold's fixtures/launches.json) for AMD gfx1151 (wave32) into
an HSA code object (.hsaco) and an aot.json manifest in the CUDA kit's schema (zig/src/cuda/aot.zig's KernelJson) plus
the AMD fields (arch, warp_size, block, file, sha256).

A variant is keyed exactly as the CUDA set keys it: the function, every constexpr (as recorded, PDL included: on AMD
gdc_wait is a no-op, so the existing Zig wrappers pass the same constexprs), each runtime parameter's type, and its
16-divisibility (pointers: 16-byte aligned; ints: value % 16 == 0 unless do_not_specialize). An int argument equal to
1 is folded into a constexpr, as Triton does and as the recording holds it.

Kernel ABI (Triton 3.8 AMD launcher, backends/amd/driver.c): the runtime parameters in declaration order (pointers 8
bytes, i32 4, fp32 4), then two pointers (global scratch, profile scratch; null when their sizes are 0); block
(num_warps * 32, 1, 1); dynamic LDS = shared.

Usage: python aot_build.py LAUNCHES_JSON OUT_DIR [--extra DERIVED_JSON] [--only fn,fn] [--jobs N]
--extra (repeatable) adds launches in the same schema that the recording lacks (variants.py's derived_launches.json:
the prompt GEMM shapes from the weights, every decode / verify window size); the manifest records their hashes.
"""

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TORCH_TO_TRITON = {"bfloat16": "*bf16", "float16": "*fp16", "float32": "*fp32", "int64": "*i64", "int32": "*i32",
                   "int16": "*i16", "uint8": "*u8", "int8": "*i8", "bool": "*i1", "float8_e4m3fn": "*u8"}
# per-function compile options: what the Python wrappers pass beyond num_warps, and AMD retunes (a kernel's warps are
# metadata, not part of the variant key: the launcher reads block = num_warps * 32 from the manifest)
EXTRA_OPTIONS = {
    "_sparse_attn_part": {"num_stages": 1},
    # 1 warp (the CUDA tiling) on gfx1151 spills 163 VGPRs and, with TREE, its tile maxima came out up to 2 ulp off
    # the keys it stored (t_index: 1444 / 8192 tiles); 2 warps: no spills, maxima exact
    "_index_score_rows": {"num_warps": 2},
}
ARCH = "gfx1151"


def const_value(v):
    if isinstance(v, dict):
        return float(v["float"])
    return v


def plan_variant(fn, case):
    """(signature, constexprs, attrs, params, consts_json) of one recorded launch."""
    sig, constexprs, attrs, params, consts_json = {}, {}, {}, [], {}
    defaults = {k: p.default for k, p in inspect.signature(fn.fn).parameters.items()
                if p.default is not inspect.Parameter.empty}
    nospec = set(fn.do_not_specialize) if fn.do_not_specialize else set()
    nospec = {fn.arg_names[i] if isinstance(i, int) else i for i in nospec}
    for i, name in enumerate(fn.arg_names):
        if name in case["consts"]:
            v = const_value(case["consts"][name])
            sig[name] = "constexpr"
            constexprs[name] = v
            raw = case["consts"][name]
            if isinstance(raw, dict):
                consts_json[name] = {"f32": int(raw["fp32_bits"], 16)}
            else:
                consts_json[name] = {"int": int(raw)}
        elif name in case["tensors"]:
            dtype, _shape, _strides, aligned = case["tensors"][name]
            ty = TORCH_TO_TRITON[dtype]
            sig[name] = ty
            if aligned:
                attrs[(i,)] = [["tt.divisibility", 16]]
            params.append({"name": name, "type": ty, "div16": bool(aligned), "nospec": False})
        elif name in case["scalars"]:
            v = case["scalars"][name]
            if isinstance(v, dict):
                sig[name] = "fp32"
                params.append({"name": name, "type": "fp32", "div16": False, "nospec": name in nospec})
            else:
                assert -2**31 <= v < 2**31, (name, v)
                sig[name] = "i32"
                ns = name in nospec
                d16 = (not ns) and v != 1 and v % 16 == 0
                if d16:
                    attrs[(i,)] = [["tt.divisibility", 16]]
                params.append({"name": name, "type": "i32", "div16": d16, "nospec": ns})
        elif name in defaults:
            sig[name] = "constexpr"
            constexprs[name] = defaults[name].value if hasattr(defaults[name], "value") else defaults[name]
        else:
            raise KeyError(f"{fn.__name__}: parameter {name} not in the recording and has no default")
    return sig, constexprs, attrs, params, consts_json


def build_one(fn_name, ci, case, out_dir):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    import tri_kernels as K

    fn = getattr(K, fn_name)
    sig, constexprs, attrs, params, consts_json = plan_variant(fn, case)
    opts = {"num_warps": case["num_warps"]}
    opts.update(EXTRA_OPTIONS.get(fn_name, {}))
    t0 = time.time()
    try:
        src = ASTSource(fn, sig, constexprs=constexprs, attrs=attrs)
        k = triton.compile(src, target=GPUTarget("hip", ARCH, 32), options=opts)
    except Exception as e:  # noqa: BLE001 - reported per variant
        return {"fn": fn_name, "case": ci, "phase": case["phase"], "ok": False,
                "error": f"{type(e).__name__}: {e}"[:4000], "trace": traceback.format_exc()[-6000:]}
    dt = time.time() - t0
    hsaco = k.asm["hsaco"]
    sha = hashlib.sha256(hsaco).hexdigest()
    md = k.metadata
    (Path(out_dir) / "hsaco").mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "hsaco" / f"{sha}.hsaco").write_bytes(hsaco)
    amdgcn = k.asm.get("amdgcn", "")
    stats = {}
    for line in amdgcn.splitlines():
        line = line.strip()
        for key in (".vgpr_count:", ".sgpr_count:", ".vgpr_spill_count:", ".sgpr_spill_count:",
                    ".private_segment_fixed_size:", ".group_segment_fixed_size:"):
            if line.startswith(key):
                stats[key.strip(".:")] = int(line.split(":")[1].strip().split()[0])
    return {
        "ok": True, "fn": fn_name, "case": ci, "phase": case["phase"], "compile_s": round(dt, 2),
        "kernel": {
            "fn": fn_name, "hash": sha, "name": md.name, "num_warps": md.num_warps, "num_ctas": md.num_ctas,
            "shared": md.shared, "global_scratch": md.global_scratch_size, "global_align": md.global_scratch_align,
            "profile_scratch": md.profile_scratch_size, "pdl": False, "params": params, "consts": consts_json,
            "arch": ARCH, "warp_size": md.warp_size, "block": md.num_warps * md.warp_size,
            "file": f"hsaco/{sha}.hsaco", "options": opts, "isa_stats": stats,
            "recorded_grid": case["grid"],
        },
    }


def toolchain(launches, extra=()):
    """The pinned compiler and the inputs the variants were built from (the manifest's toolchain record)."""
    import platform
    from importlib.metadata import version
    src = HERE / "tri_kernels.py"
    wheel = os.environ.get("TRI_TRITON_WHEEL_SHA256", "1bab2eeb8f4beccad389a497a45b3018fa6291ab93c577f04b1376cf12422499")
    v = version("triton")
    return {"version": v, "wheel": f"triton-{v}-cp313-cp313-linux_x86_64.whl",
            "wheel_sha256": wheel, "index": "https://rocm.nightlies.amd.com/whl-multi-arch/",
            "python": platform.python_version(), "target": "hip:gfx1151:32",
            "kernels_py_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "launches_json_sha256": hashlib.sha256(Path(launches).read_bytes()).hexdigest(),
            "extra_launches": [{"file": Path(e).name, "sha256": hashlib.sha256(Path(e).read_bytes()).hexdigest()}
                               for e in extra],
            "extra_options": EXTRA_OPTIONS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("launches")
    ap.add_argument("out")
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    launches = json.loads(Path(a.launches).read_text())
    for e in a.extra:
        for fn, cases in json.loads(Path(e).read_text()).items():
            launches.setdefault(fn, []).extend(cases)
    only = set(filter(None, a.only.split(",")))
    jobs = [(fn, i, c) for fn, cases in sorted(launches.items()) if not only or fn in only
            for i, c in enumerate(cases)]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(build_one, fn, i, c, a.out) for fn, i, c in jobs]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            tag = "ok " if r["ok"] else "ERR"
            extra = (f"{r['compile_s']}s vgpr={r['kernel']['isa_stats'].get('vgpr_count')} "
                     f"spill={r['kernel']['isa_stats'].get('vgpr_spill_count')} lds={r['kernel']['shared']}"
                     if r["ok"] else r["error"].splitlines()[0][:200])
            print(f"{tag} {r['fn']}[{r['case']}] {r['phase']}: {extra}", flush=True)
    results.sort(key=lambda r: (r["fn"], r["case"]))
    kernels = []
    seen = set()
    for r in results:
        if r["ok"]:
            key = (r["fn"], json.dumps(r["kernel"]["params"], sort_keys=True), json.dumps(r["kernel"]["consts"], sort_keys=True),
                   json.dumps(r["kernel"]["options"], sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            kernels.append(r["kernel"])
    tc = toolchain(a.launches, a.extra)
    (Path(a.out) / "aot.json").write_text(json.dumps({"arch": ARCH, "triton": tc["version"], "triton_build": tc,
                                                      "kernels": kernels}, indent=1))
    (Path(a.out) / "build_log.json").write_text(json.dumps(results, indent=1))
    bad = [r for r in results if not r["ok"]]
    print(f"variants: {len(results)} built {len(results) - len(bad)} failed {len(bad)}; distinct {len(kernels)}")
    for r in bad:
        print(f"FAILED {r['fn']}[{r['case']}]: {r['error'][:300]}")


if __name__ == "__main__":
    main()
