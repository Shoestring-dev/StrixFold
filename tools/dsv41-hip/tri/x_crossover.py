"""Where the prompt-path HIP kernels overtake the Triton ones as the row count grows (the wrapper's dispatch rule):
aux.index_rows vs _index_score_rows (16K keys, keys + tile maxima) and aux.hc_mix_pf vs _hc_mix_pf (posted, world 2),
1..512 rows. HIP events, interleaved rounds."""
import json, statistics
import numpy as np
import common as C, hiprt as H, t_index as TI, t_hipindex as TX, t_hc as TH, t_hiphc as TC
from hiprt import Buf, P
C.log_open("x_crossover")
L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
rng = np.random.default_rng(11)
C.gpu_lock()
kr = TI.consts_of(next(c for c in L["_index_score_rows"] if c["consts"]["KEYS"] and c["consts"]["HAS_TMAX"] and "nt" not in c["consts"]))
kpf = TH.consts_of(L["_hc_mix_pf"][1])
n, D = 16384, 5120
nt = n // 64
codes = Buf.of(rng.integers(0, 256, (n, 64), dtype=np.uint8))
scales = Buf.of(np.full((n, 4), 120, np.uint8))
setups = []
for rows in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    q, w = Buf.of(C.rand_bf16(rng, (rows, 32, 128), 0.5)), Buf.of(C.rand_bf16(rng, (rows, 32), 0.2))
    vis = Buf.of(np.full(rows, n, np.int64))
    out, tm = Buf(rows * n * 8), Buf(rows * nt * 8)
    ia = TX.IndexArgs(q=q.ptr, k=codes.ptr, ks=scales.ptr, w=w.ptr, vis=vis.ptr, out=out.ptr, tmax=tm.ptr, rows=rows, n=n, nt=nt, flags=3)
    targs = dict(Q=P(q, "*bf16"), K=P(codes, "*u8"), KS=P(scales, "*u8"), Wt=P(w, "*bf16"), VIS=P(vis, "*i64"),
                 OUT=P(out, "*i64"), n=n, rows=rows, TMAX=P(tm, "*i64"), nt=nt)
    x = Buf.of(C.rand_bf16(rng, (rows, 4, D), 1.0))
    g = Buf.of((rng.standard_normal((2, rows, D)) * 0.5).astype(np.float32))
    post, comb = Buf.of((2 * rng.random((rows, 4))).astype(np.float32)), Buf.of(rng.random((rows, 4, 4)).astype(np.float32))
    xo, part = Buf(rows * 4 * D * 2), Buf(rows * 40 * 32 * 4)
    ha = TC.HcArgs(x=x.ptr, xo=xo.ptr, g=g.ptr, post=post.ptr, comb=comb.ptr, fn=TH.FN_B.ptr, part=part.ptr, rs=rows * D,
                   rows=rows, world=2, flags=1)
    hargs = dict(X=P(x, "*bf16"), XO=P(xo, "*bf16"), G=P(g, "*fp32"), RS=rows * D, POST=P(post, "*fp32"),
                 COMB=P(comb, "*fp32"), FN=P(TH.FN_B, "*fp32"), PART=P(part, "*fp32"), rows=rows)
    setups.append((rows, ia, targs, ha, hargs, [q, w, vis, out, tm, x, g, post, comb, xo, part]))
print("load", C.quiet_window("w-tri-cross"), flush=True)
for rows, ia, targs, ha, hargs, _ in setups:
    t = {k: [] for k in ("idx_hip", "idx_tri", "hc_hip", "hc_tri")}
    for _ in range(5):
        t["idx_hip"].append(H.time_ms(lambda: H.launch(TX.FN, ((rows + TX.RT - 1) // TX.RT, TX.splits_for(rows, nt), 1), (256, 1, 1), 0, [ia]), iters=3, warmup=1))
        t["idx_tri"].append(H.time_ms(lambda: C.KIT.run("_index_score_rows", (rows, nt, 1), targs, kr), iters=3, warmup=1))
        t["hc_hip"].append(H.time_ms(lambda: H.launch(TC.FN, (((rows + TC.RB - 1) // TC.RB) * 10, 1, 1), (256, 1, 1), 0, [ha]), iters=3, warmup=1))
        k2 = dict(kpf)
        t["hc_tri"].append(H.time_ms(lambda: C.KIT.run("_hc_mix_pf", (((rows + 7) // 8) * 40, 1, 1), hargs, k2), iters=3, warmup=1))
    med = {k: statistics.median(v) for k, v in t.items()}
    print(f"rows={rows:4d}  index hip {med['idx_hip']*1e3:8.1f} us  triton {med['idx_tri']*1e3:8.1f} us   "
          f"hc_mix hip {med['hc_hip']*1e3:7.1f} us  triton {med['hc_tri']*1e3:7.1f} us", flush=True)
    C.record(kernel="crossover", case=f"rows={rows}", ok=None, **med)
C.done()
