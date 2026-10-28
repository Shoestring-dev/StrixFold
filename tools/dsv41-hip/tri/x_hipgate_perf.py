"""Device time of the HIP router gate (aux.gate_logits) against the engine's current prompt path, hipBLAS sgemm on fp32
copies of x and the weights (the toF32 / f16ToF32 conversions not included), 384 experts (and 128, the drafter's),
512-8192 rows. Launches only; run under rocprofv3 --kernel-trace (this process takes the GPU lock and a quiet window
itself) and read the runs with x_runs.py."""
import numpy as np
import common as C, hiprt as H, t_hipgate as T
from hiprt import Buf
C.log_open("x_hipgate_perf")
rng = np.random.default_rng(12)
D = 5120
C.gpu_lock()
blas = T.HipBlas()
cases = []
for rows, ne in ((512, 384), (2048, 384), (8192, 384), (2048, 128), (512, 32), (2048, 32), (8192, 32)):
    x = C.rand_bf16(rng, (rows, D), 1.0)
    w = (rng.standard_normal((ne, D)) * 0.02).astype(np.float16)
    cases.append((rows, ne, x, w))
C.record(kernel="window", case="quiet", ok=None, load=C.quiet_window("w-tri-gate"))
for rows, ne, x, w in cases:
    xb, wb, out = Buf.of(x), Buf.of(w), Buf(rows * ne * 4)
    xf, wf = Buf.of(C.from_bf16_bits(x).astype(np.float32)), Buf.of(w.astype(np.float32))
    a = T.GateArgs(x=xb.ptr, w=wb.ptr, out=out.ptr, rows=rows, ne=ne, flags=0)
    C.record(kernel="case", case=f"rows={rows} ne={ne} hip", ok=None)
    for _ in range(5):
        f, bm, bn, nt = T.build_of(ne)
        H.launch(f, ((rows + bm - 1) // bm, ne // bn, 1), (nt, 1, 1), 0, [a])
    H.sync()
    C.record(kernel="case", case=f"rows={rows} ne={ne} hipblas", ok=None)
    for _ in range(5):
        blas.xwT(xf.ptr, wf.ptr, out.ptr, rows, D, ne)
    H.sync()
C.done()
