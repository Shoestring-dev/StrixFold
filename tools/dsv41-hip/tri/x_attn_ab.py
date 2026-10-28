"""A/B timing of aux_attn.hip builds on one set of inputs, rounds interleaved so that every build sees the same box
state (the GPU shares DRAM and power with the CPU jobs). Usage: x_attn_ab.py name:block:hsaco ... (HIP events)."""
import statistics
import sys

import common as C
import hiprt as H
import t_hipattn as T
import x_hipattn_perf as XP

C.log_open("x_attn_ab")
builds = []
for spec in sys.argv[1:]:
    name, block, path = spec.split(":", 2)
    builds.append((name, int(block), H.Module(open(path, "rb").read()).function("tf_ds_aux_sparse_attn_h32_d512")))
C.gpu_lock()
cases = []
for rows, final, corr in ((16, False, False), (512, True, False), (2048, True, False), (2048, True, True)):
    d = XP.inputs(rows, final, corr)
    keep, captured = [], {}
    real = H.launch
    def grab(f, grid, block, shared, args):
        captured["a"], captured["grid"] = args[0], grid
        return real(f, grid, block, shared, args)
    H.launch = grab
    T.hip_attn(d, 1 if final else 8, keep)
    H.launch = real
    cases.append((f"rows={rows} final={final}" + (" corr" if corr else ""), captured["a"], captured["grid"], keep))
load = C.quiet_window("w-tri-attn-ab")
print("load", load, flush=True)
for tag, a, grid, _ in cases:
    times = {n: [] for n, _, _ in builds}
    for _ in range(7):
        for n, block, f in builds:
            times[n].append(H.time_ms(lambda: H.launch(f, grid, (block, 1, 1), 0, [a]), iters=3, warmup=1))
    line = "  ".join(f"{n} {statistics.median(v):7.3f}" for n, v in times.items())
    print(f"{tag:22s} {line}", flush=True)
    C.record(kernel="aux.sparse_attn", case=tag + " A/B", ok=None, load=load, **{n: statistics.median(v) for n, v in times.items()})
C.done()
