"""The prompt indexer's top-k selection at 2048 rows: aux.topk_i64 (topk_select's torch step, ops.topkI64) over n keys
for n = 8K..32K, and _topk_finish (sorting the indices); the pruned path (n > 32K) adds a top-k over the tile maxima
and _gather_tiles, then the same top-k over k x 64 candidates. HIP events."""
import ctypes, json, os
import numpy as np
import common as C, hiprt as H
from hiprt import Buf, P
C.log_open("x_topk_perf")
L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
FN = H.Module(open(os.environ.get("TRI_AUX_HSACO", str(C.ROOT / "build" / "aux.hsaco")), "rb").read()).function("tf_ds_aux_topk_i64")
rng = np.random.default_rng(4)
rows, k = 2048, 512
C.gpu_lock()
kf = next({kk: (v if not isinstance(v, dict) else float(v["float"])) for kk, v in c["consts"].items()} for c in L["_topk_finish"]
          if c["consts"]["KK"] == k)
setups = []
for n in (1024, 8192, 16384, 32768, 65536):
    sc = rng.standard_normal((rows, n)).astype(np.float32)          # indexer-like keys: score bits over ~index
    b = sc.view(np.int32).astype(np.int64)
    keys = Buf.of((np.where(b < 0, b ^ 0x7FFFFFFF, b) << 32) | (0xFFFFFFFF - np.arange(n, dtype=np.int64))[None, :])
    top, out, vis = Buf(rows * k * 8), Buf(rows * k * 8), Buf.of(np.full(rows, n, np.int64))
    setups.append((n, keys, top, out, vis))
print("load", C.quiet_window("w-tri-topk"), flush=True)
for n, keys, top, out, vis in setups:
    args = [ctypes.c_uint64(keys.ptr), ctypes.c_int64(n), ctypes.c_int32(n), ctypes.c_int32(k), ctypes.c_uint64(top.ptr)]
    ms = H.time_ms(lambda: H.launch(FN, (rows, 1, 1), (1024, 1, 1), 0, args), iters=5, warmup=1)
    fa = dict(TOP=P(top, "*i64"), VIS=P(vis, "*i64"), OUT=P(out, "*i64"))
    mf = H.time_ms(lambda: C.KIT.run("_topk_finish", (rows, 1, 1), fa, kf), iters=5, warmup=1)
    print(f"n={n:6d} topk_i64 {ms:7.3f} ms  _topk_finish {mf:6.3f} ms", flush=True)
    C.record(kernel="topk", case=f"rows={rows} n={n} k={k}", ok=None, topk_i64_ms=ms, finish_ms=mf)
C.done()
