"""Device time of the HIP heads RoPE (aux.rope_heads) against the Triton _rope_heads: H = 32, HD = 512 (q, o) and 128
(the indexer's q), 1-8192 rows. HIP events, interleaved rounds."""
import json, statistics
import numpy as np
import common as C, hiprt as H, t_hiprope as T
from hiprt import Buf, P
C.log_open("x_hiprope_perf")
L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
rng = np.random.default_rng(9)
C.gpu_lock()
cos = Buf.of(rng.standard_normal((T.NPOS, 32)).astype(np.float32))
setups = []
for hd in (512, 128):
    k = next({kk: v for kk, v in c["consts"].items()} for c in L["_rope_heads"]
             if c["consts"]["HD"] == hd and c["consts"]["H"] == 32 and not c["consts"]["INV"])
    for rows in (1, 6, 2048, 8192):
        x, pos = Buf.of(C.rand_bf16(rng, (rows, 32, hd), 1.0)), Buf.of(rng.integers(0, T.NPOS, rows).astype(np.int64))
        a = T.RopeArgs(x=x.ptr, cos=cos.ptr, sin=cos.ptr, pos=pos.ptr, rows=rows, h=32, hd=hd, rd=64, flags=0)
        targs = dict(X=P(x, "*bf16"), COS=P(cos, "*fp32"), SIN=P(cos, "*fp32"), POS=P(pos, "*i64"))
        setups.append((hd, rows, k, a, targs, x, pos))
print("load", C.quiet_window("w-tri-rope"), flush=True)
for hd, rows, k, a, targs, _, _ in setups:
    n = rows * 32 * 8
    th, tt = [], []
    for _ in range(7):
        th.append(H.time_ms(lambda: H.launch(T.FN, ((n + 255) // 256, 1, 1), (256, 1, 1), 0, [a]), iters=20, warmup=2))
        tt.append(H.time_ms(lambda: C.KIT.run("_rope_heads", (rows, 32 // k["HB"], 1), targs, k), iters=20, warmup=2))
    print(f"HD={hd} rows={rows:5d}  hip {statistics.median(th) * 1e3:7.1f} us  triton {statistics.median(tt) * 1e3:7.1f} us", flush=True)
    C.record(kernel="aux.rope_heads", case=f"HD={hd} rows={rows}", ok=None, hip_us=statistics.median(th) * 1e3,
             triton_us=statistics.median(tt) * 1e3)
C.done()
