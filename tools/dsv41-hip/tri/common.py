"""Test plumbing: the kit, metrics against the float64 references, a results log (results/*.jsonl)."""

import json
import os
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

import hiprt as H

_GPU_LOCK = None


def gpu_lock(timeout=1800):
    """Take /var/tmp/halo-gpu.lock in this process, for this run (the house rule: the GPU lock per run, never a
    sweep; halo-cpu-guard spares lock holders and their children during quiet windows)."""
    global _GPU_LOCK
    if _GPU_LOCK is not None:
        return
    import fcntl
    import signal
    fd = open("/var/tmp/halo-gpu.lock", "a")

    def expired(_sig, _frame):
        raise SystemExit("halo-gpu.lock: timed out")

    # a blocking wait, as flock(1) waits (a non-blocking poll loses the lock to blocking waiters every time)
    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(timeout)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    _GPU_LOCK = fd


def quiet_window(tag):
    """A timing measurement's quiet window (~/bin/halo-quiet on/off, closed at exit), under the GPU lock; inputs are
    prepared before it. Returns the load average at the start (rerun anything measured above 2)."""
    import atexit
    import subprocess
    gpu_lock()
    q = str(Path.home() / "bin" / "halo-quiet")
    subprocess.run([q, "on", tag], check=False)
    atexit.register(lambda: subprocess.run([q, "off", tag], check=False))
    return open("/proc/loadavg").read().split()[:3]


if os.environ.get("TRI_GPU_LOCK") == "1":
    gpu_lock()

# the work area: upstream/ (pinned inputs), kit/, results/, logs/ (default node1:~/tfp/tri; TRI_WORK overrides)
ROOT = Path(os.environ.get("TRI_WORK", Path.home() / "tfp" / "tri"))
KIT = H.Kit(os.environ.get("TRI_KIT", ROOT / "kit" / "gfx1151"))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def launches():
    """The kit's launches: the recording (upstream/launches.json) and, after each function's recorded cases, the
    derived ones the recording lacks (zig/src/families/dsv41/fixtures/derived_launches.json, variants.py: prompt GEMM
    shapes, window sizes)."""
    rec = json.loads((ROOT / "upstream" / "launches.json").read_text())
    derived = Path(__file__).resolve().parents[3] / "zig" / "src" / "families" / "dsv41" / "fixtures" / "derived_launches.json"
    for fn, cases in json.loads(derived.read_text()).items():
        rec.setdefault(fn, []).extend(cases)
    return rec
_log = None
FAILS = []


def log_open(name):
    global _log
    _log = open(RESULTS / f"{name}.jsonl", "a")


def record(**kw):
    kw["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = json.dumps(kw, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if _log:
        _log.write(line + "\n")
        _log.flush()
    ok = kw.get("ok")
    tag = "PASS" if ok else ("FAIL" if ok is False else "INFO")
    brief = {k: v for k, v in kw.items() if k not in ("time", "ok", "kernel", "case")}
    print(f"[{tag}] {kw.get('kernel')} {kw.get('case', '')}: {json.dumps(brief, default=str)[:600]}", flush=True)
    if ok is False:
        FAILS.append((kw.get("kernel"), kw.get("case")))


def bf16_bits(x):
    return np.asarray(x, np.float64).astype(ml_dtypes.bfloat16).view(np.uint16)


def from_bf16_bits(b):
    return np.asarray(b, np.uint16).view(ml_dtypes.bfloat16).astype(np.float64)


def rand_bf16(rng, shape, scale=1.0):
    """Random bf16 values (bits), normal * scale, as stored tensors are."""
    return bf16_bits(rng.standard_normal(shape) * scale)


def ulp_bf16(ref):
    """bf16 ulp at |ref| (subnormals: 2^-133)."""
    a = np.abs(np.asarray(ref, np.float64))
    e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    return np.where(a > 0, np.ldexp(1.0, (np.maximum(e, -126) - 7).astype(np.int64)), 2.0 ** -133)


def ulp_f16(ref):
    a = np.abs(np.asarray(ref, np.float64))
    e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    return np.where(a > 0, np.ldexp(1.0, (np.maximum(e, -14) - 10).astype(np.int64)), 2.0 ** -24)


def cmp_rounded(got, ref_exact, rounded_ref, ulp_fn, max_ulp=1, max_frac=1e-2, bound=None):
    """A stored (bf16/fp16) output against the float64 reference: `ref_exact` the fp64 value before the output's
    rounding, `rounded_ref` its correctly rounded value.

    Without `bound`: every element within max_ulp output ulps of the exact value. With `bound` (an elementwise
    forward-error bound of the fp32 computation before the output rounding, e.g. gamma_m * sum|terms|): every element
    within bound + half an output ulp at |exact| + bound. Either way at most max_frac of the elements may differ from
    the correctly rounded value (a rounding flip needs the fp32 value within its error of a rounding boundary)."""
    got = np.asarray(got, np.float64)
    ref_exact = np.asarray(ref_exact, np.float64)
    nonfinite = int((~np.isfinite(got)).sum())
    err = np.abs(got - ref_exact)
    ulps = err / ulp_fn(ref_exact)
    mism = float(np.mean(got != rounded_ref))
    nl2 = float(np.linalg.norm(got - ref_exact) / max(np.linalg.norm(ref_exact), 1e-300))
    m = dict(max_ulp=float(np.nanmax(ulps)) if ulps.size else 0.0, frac_not_nearest=mism,
             max_abs=float(np.nanmax(err)) if got.size else 0.0, nl2=nl2, nonfinite=nonfinite, n=int(got.size))
    if bound is None:
        within = m["max_ulp"] <= max_ulp + 0.5
    else:
        bound = np.broadcast_to(np.asarray(bound, np.float64), err.shape)
        allow = bound + 0.5 * ulp_fn(np.abs(ref_exact) + bound) * (1 + 2.0 ** -6)
        r = err / allow
        m["max_err_over_allowed"] = float(np.nanmax(r)) if r.size else 0.0
        within = m["max_err_over_allowed"] <= 1.0
    m["ok"] = bool(nonfinite == 0 and within and mism <= max_frac)
    return m


def cmp_bound(got, ref, bound):
    """An fp32 output against fp64 with an elementwise forward-error bound (gamma_m * sum|terms| etc.)."""
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    err = np.abs(got - ref)
    ratio = err / np.maximum(bound, 1e-300)
    nonfinite = int((~np.isfinite(got)).sum())
    m = dict(max_abs=float(err.max()) if err.size else 0.0, max_err_over_bound=float(ratio.max()) if err.size else 0.0,
             nl2=float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-300)), nonfinite=nonfinite,
             n=int(got.size))
    m["ok"] = bool(nonfinite == 0 and m["max_err_over_bound"] <= 1.0)
    return m


def gamma(m, u=2.0 ** -24):
    return m * u / (1 - m * u)


_EXTRA = {}


def extra_variant(fn, base_case, **overrides):
    """A test-only variant (not in the recorded set): base_case's specialization with constexprs overridden, compiled
    into kit/test-extra by the kit builder's own code."""
    import copy
    import aot_build
    case = copy.deepcopy(base_case)
    for k, v in overrides.items():
        case["consts"][k] = v
    key = (fn, json.dumps(case["consts"], sort_keys=True), json.dumps(case["tensors"], sort_keys=True), case["num_warps"])
    if key not in _EXTRA:
        out = ROOT / "kit" / "test-extra"
        r = aot_build.build_one(fn, 0, case, str(out))
        if not r["ok"]:
            raise RuntimeError(r["error"])
        r["kernel"]["file"] = str(out / r["kernel"]["file"])
        _EXTRA[key] = r["kernel"]
    return _EXTRA[key]


def jit_kernel(fn, signature, constexprs, num_warps=4):
    """A test-only @triton.jit kernel (not a kit variant) compiled for gfx1151 by the kit's Triton: (function, block
    threads, LDS bytes, module to keep alive). hiprt.launch it with the runtime arguments, then two null scratch
    pointers."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    import aot_build
    k = triton.compile(ASTSource(fn, signature, constexprs=constexprs), target=GPUTarget("hip", aot_build.ARCH, 32),
                       options={"num_warps": num_warps})
    m = H.Module(k.asm["hsaco"])
    return m.function(k.metadata.name), k.metadata.num_warps * k.metadata.warp_size, k.metadata.shared, m


def run_exact(fn, grid, args, consts, tag):
    """fn on the kit's own variant for this launch (exact selection, as TF_GPU_STRICT=1); False, with the failure
    recorded, when the kit has none."""
    try:
        var = KIT.find(fn, args, consts, exact_only=True)
    except KeyError as e:
        record(kernel=fn, case=f"{tag}: the kit's exact variant", ok=False, error=str(e)[:300])
        return False
    KIT.run(fn, grid, args, consts, variant=var)
    return True


def run_variant(k, grid, args):
    """Launch kernel dict k (from extra_variant) with its own file path."""
    import ctypes
    if k["hash"] not in KIT._loaded:
        m = H.Module(Path(k["file"]).read_bytes())
        KIT._loaded[k["hash"]] = (m, m.function(k["name"]))
    kk = dict(k)
    kk["file"] = k["file"]
    return KIT.run(k["fn"], grid, args, {}, variant=kk)


def done():
    if _log:
        _log.close()
    print(f"failures: {len(FAILS)} {FAILS[:20]}")
    return 1 if FAILS else 0
