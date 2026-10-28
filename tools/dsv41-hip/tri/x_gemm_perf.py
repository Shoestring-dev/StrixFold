"""Device time of the prompt GEMM _gemm on every kit shape (recorded and derived) at M rows, its kit build (8 warps)
against other compile options (num_warps / num_stages are kit metadata, not variant keys: the engine launches
block = num_warps * 32 from the manifest), the builds' rounds interleaved (the GPU shares DRAM and power with the CPU).
This process takes the GPU lock and a quiet window itself; it prints the load and the GPU clock with the numbers.

Usage: x_gemm_perf.py [M] [warps:stages[:BM:BK] ...] [--shapes K/N,...]   (default 2048 8:2 16:2; BM and BK are
variant keys the wrapper sets, 128 and 32 in the served lane: other values are test builds of a tiling the Zig wrapper
would have to choose)"""
import copy
import sys
from pathlib import Path

import numpy as np

import aot_build
import common as C
import hiprt as H
from hiprt import Buf, P

argv = [a for a in sys.argv[1:] if not a.startswith("--shapes")]
SHAPES = {tuple(int(v) for v in x.split("/")) for a in sys.argv[1:] if a.startswith("--shapes=")
          for x in a.split("=", 1)[1].split(",")}
M = int(argv[0]) if argv else 2048
OPTS = [tuple(int(v) for v in a.split(":")) + (128, 32)[len(a.split(":")) - 2:] for a in argv[1:]] or [(8, 2, 128, 32),
                                                                                                      (16, 2, 128, 32)]
OUT = C.ROOT / "kit" / "test-extra"
rng = np.random.default_rng(31)
C.log_open("x_gemm_perf")


def sclk():
    for f in Path("/sys/class/drm").glob("card*/device/pp_dpm_sclk"):
        cur = [ln for ln in f.read_text().splitlines() if ln.endswith("*")]
        return cur[0] if cur else "?"
    return "?"


builds = []
for ci, c in enumerate(C.launches()["_gemm"]):
    if SHAPES and (c["consts"]["K"], c["consts"]["N"]) not in SHAPES:
        continue
    vs = []
    for nw, ns, bm, bk in OPTS:
        cc = copy.deepcopy(c)
        cc["num_warps"] = nw
        cc["consts"]["BM"], cc["consts"]["BK"] = bm, bk
        saved = aot_build.EXTRA_OPTIONS.get("_gemm")
        aot_build.EXTRA_OPTIONS["_gemm"] = {"num_stages": ns}
        r = aot_build.build_one("_gemm", ci, cc, str(OUT))
        if saved is None:
            aot_build.EXTRA_OPTIONS.pop("_gemm")
        if not r["ok"]:
            print(f"K={c['consts']['K']} N={c['consts']['N']} w{nw}s{ns}m{bm}k{bk}: {r['error'][:200]}", flush=True)
            continue
        st = r["kernel"]["isa_stats"]
        print(f"K={c['consts']['K']} N={c['consts']['N']} w{nw}s{ns}m{bm}k{bk}: vgpr {st.get('vgpr_count')} spill "
              f"{st.get('vgpr_spill_count')} lds {r['kernel']['shared']}", flush=True)
        kd = r["kernel"]
        kd["file"] = str(OUT / kd["file"])
        vs.append(((nw, ns, bm, bk), kd))
    builds.append((c, vs))

C.gpu_lock()
cases = []
for c, vs in builds:
    k, n = c["consts"]["K"], c["consts"]["N"]
    os_ = c["scalars"]["o_stride"]
    ob = 4 if c["tensors"]["OUT"][0] == "float32" else 2
    x = Buf.of(rng.standard_normal((M, k)).astype(np.float16))
    w = Buf.of(rng.standard_normal((k, n)).astype(np.float16))
    h = Buf.of(np.ones((128, 128), np.uint16) * 0x3F80)
    sv = Buf.of((rng.standard_normal(n) * 0.02).astype(np.float16))
    out = Buf(M * os_ * ob)
    args = dict(X=P(x, "*fp16"), W=P(w, "*fp16"), H=P(h, "*bf16"), SVH=P(sv, "*fp16"), BIAS=P(sv, "*fp16"),
                OUT=P(out, "*fp32" if ob == 4 else "*bf16"), M=M, o_stride=os_)
    cases.append((c, vs, args, (x, w, h, sv, out)))
load = C.quiet_window("ktri-gemm")
print(f"load {load} sclk {sclk()} M={M}", flush=True)
for c, vs, args, _keep in cases:
    k, n, dt = c["consts"]["K"], c["consts"]["N"], c["tensors"]["OUT"][0]
    times = {o: [] for o, _ in vs}
    for _ in range(5):
        for o, kd in vs:
            grid = (M + o[2] - 1) // o[2] * (n // 128)
            times[o].append(H.time_ms(lambda kd=kd, g=grid: C.run_variant(kd, (g, 1, 1), args), iters=5, warmup=1))
    flop = 2.0 * M * k * n + 2.0 * 2 * M * n * 128
    line = "  ".join(f"w{o[0]}s{o[1]}m{o[2]}k{o[3]} {np.median(t):.3f} ms ({flop / np.median(t) / 1e9:.1f} TF/s)"
                     for o, t in times.items())
    print(f"K={k:5d} N={n:5d} {dt:8s} {line}", flush=True)
    C.record(kernel="_gemm", case=f"M={M} K={k} N={n} {dt}", ok=None,
             ms={f"w{o[0]}s{o[1]}m{o[2]}k{o[3]}": float(np.median(t)) for o, t in times.items()})
print(f"sclk after {sclk()} load {open('/proc/loadavg').read().split()[:3]}", flush=True)
C.done()
