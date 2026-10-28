"""Device time of the HIP prompt indexer against the Triton _index_score_rows (keys + tile maxima) for 2048-row
chunks at 8K-64K compressed keys (16K-128K context at ratio 2), all keys visible. Launches only (no downloads); run
under rocprofv3 --kernel-trace (this process takes the GPU lock and a quiet window itself) and read the runs with
x_runs.py."""
import json
import numpy as np
import common as C, hiprt as H, t_index as TI, t_hipindex as T
from hiprt import Buf, P
L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
C.log_open("x_hipindex_perf")
rng = np.random.default_rng(6)
kr = TI.consts_of(next(c for c in L["_index_score_rows"] if c["consts"]["KEYS"] and c["consts"]["HAS_TMAX"] and "nt" not in c["consts"]))
C.gpu_lock()
cases = []
for rows, n in ((512, 8192), (2048, 8192), (2048, 16384), (2048, 32768), (2048, 65536)):
    codes = rng.integers(0, 256, (n, 64), dtype=np.uint8)
    scales = np.full((n, 4), 120, np.uint8)
    q, w = C.rand_bf16(rng, (rows, 32, 128), 0.5), C.rand_bf16(rng, (rows, 32), 0.2)
    cases.append((rows, n, codes, scales, q, w, np.full(rows, n, np.int64)))
C.record(kernel="window", case="quiet", ok=None, load=C.quiet_window("w-tri-idx"))
for rows, n, codes, scales, q, w, vis in cases:
    nt = n // 64
    b = dict(q=Buf.of(q), k=Buf.of(codes), ks=Buf.of(scales), w=Buf.of(w), vis=Buf.of(vis))
    out, tmb = Buf(rows * n * 8), Buf(rows * nt * 8)
    a = T.IndexArgs(q=b["q"].ptr, k=b["k"].ptr, ks=b["ks"].ptr, w=b["w"].ptr, vis=b["vis"].ptr, out=out.ptr,
                    tmax=tmb.ptr, rows=rows, n=n, nt=nt, flags=3)
    C.record(kernel="case", case=f"rows={rows} n={n} hip")
    for _ in range(3):
        H.launch(T.FN, ((rows + T.RT - 1) // T.RT, T.splits_for(rows, nt), 1), (256, 1, 1), 0, [a])
    H.sync()
    C.record(kernel="case", case=f"rows={rows} n={n} triton")
    args = dict(Q=P(b["q"], "*bf16"), K=P(b["k"], "*u8"), KS=P(b["ks"], "*u8"), Wt=P(b["w"], "*bf16"),
                VIS=P(b["vis"], "*i64"), OUT=P(out, "*i64"), n=n, rows=rows, TMAX=P(tmb, "*i64"), nt=nt)
    for _ in range(3):
        C.KIT.run("_index_score_rows", (rows, nt, 1), args, kr)
    H.sync()
    del out, tmb, b
C.done()
