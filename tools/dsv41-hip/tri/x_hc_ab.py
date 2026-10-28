"""A/B timing of aux_hc.hip builds (posted, world 2) on one input set, rounds interleaved.
Usage: x_hc_ab.py name:rows_per_workgroup:hsaco ... (HIP events; grid 10 ceil(rows / RB))."""
import statistics, sys
import numpy as np
import common as C, hiprt as H, t_hc as TH, t_hiphc as T
from hiprt import Buf
C.log_open("x_hc_ab")
builds = [(s.split(":")[0], int(s.split(":")[1]), H.Module(open(s.split(":", 2)[2], "rb").read()).function("tf_ds_aux_hc_mix_pf_d5120"))
          for s in sys.argv[1:]]
rng = np.random.default_rng(9)
D = 5120
C.gpu_lock()
cases = []
for rows in (512, 2048):
    b = [Buf.of(C.rand_bf16(rng, (rows, 4, D), 1.0)), Buf.of((rng.standard_normal((2, rows, D)) * 0.5).astype(np.float32)),
         Buf.of((2 * rng.random((rows, 4))).astype(np.float32)), Buf.of(rng.random((rows, 4, 4)).astype(np.float32)),
         Buf(rows * 4 * D * 2), Buf(rows * 40 * 32 * 4)]
    a = T.HcArgs(x=b[0].ptr, xo=b[4].ptr, g=b[1].ptr, post=b[2].ptr, comb=b[3].ptr, fn=TH.FN_B.ptr, part=b[5].ptr,
                 rs=rows * D, rows=rows, world=2, flags=1)
    cases.append((rows, a, b))
print("load", C.quiet_window("w-tri-hc-ab"), flush=True)
for rows, a, _ in cases:
    t = {n: [] for n, _, _ in builds}
    for _ in range(7):
        for n, RB, f in builds:
            t[n].append(H.time_ms(lambda: H.launch(f, (((rows + RB - 1) // RB) * 10, 1, 1), (256, 1, 1), 0, [a]), iters=5, warmup=1))
    print(f"rows={rows:5d} " + "  ".join(f"{n} {statistics.median(v):6.3f}" for n, v in t.items()), flush=True)
C.done()
