"""hipBLAS sgemm for the indexer's head weights (x fp32 [rows, 5120] . idx_proj fp32 [32, 5120]^T), the prompt path's
second hipBLAS call (prompt.zig indexer): HIP events."""
import numpy as np
import common as C, hiprt as H, t_hipgate as T
from hiprt import Buf
C.log_open("x_blas32")
rng = np.random.default_rng(3)
C.gpu_lock()
blas = T.HipBlas()
setups = []
for rows in (512, 2048, 8192):
    xf = Buf.of(rng.standard_normal((rows, 5120)).astype(np.float32))
    wf = Buf.of((rng.standard_normal((32, 5120)) * 0.02).astype(np.float32))
    out = Buf(rows * 32 * 4)
    setups.append((rows, xf, wf, out))
print("load", C.quiet_window("w-tri-blas32"), flush=True)
for rows, xf, wf, out in setups:
    ms = H.time_ms(lambda: blas.xwT(xf.ptr, wf.ptr, out.ptr, rows, 5120, 32), iters=20, warmup=3)
    print(f"rows={rows} idx_proj sgemm N=32: {ms * 1e3:.1f} us", flush=True)
    C.record(kernel="hipblas idx_proj", case=f"rows={rows}", ok=None, us=ms * 1e3)
C.done()
