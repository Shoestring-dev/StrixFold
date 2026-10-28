"""Device time of the HIP sparse attention against the Triton kernel at decode / verify windows (1-16 rows, 8 splits,
ring window + 512 FP4 picks) and prompt chunks (512 / 2048 rows, one pass, linear window + 512 FP4 picks). Run under
rocprofv3 --kernel-trace (this process takes the GPU lock and a quiet window itself); the log names each case in
launch order; x_runs.py prints the runs."""

import json

import numpy as np

import common as C
import hiprt as H
import t_hipattn as T
from hiprt import Buf, P

L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
rng = np.random.default_rng(8)
C.log_open("x_hipattn_perf")
HD, Hh = 512, 32


def correlated_picks(rows, npool, keep=0.9):
    """Each row's 512 picks keep ~90% of the previous row's (as the indexer's top-k of neighbouring queries do) and
    draw the rest anew; the random picks of inputs() are the cache-unfriendly worst case."""
    out = np.empty((rows, 512), np.int64)
    cur = rng.choice(npool, 512, replace=False)
    for r in range(rows):
        if r:
            drop = rng.random(512) > keep
            fresh = rng.choice(npool, int(drop.sum()) + 64, replace=False)
            fresh = fresh[~np.isin(fresh, cur)][:int(drop.sum())]
            cur = cur.copy()
            cur[np.flatnonzero(drop)[:len(fresh)]] = fresh
        out[r] = np.sort(cur)
    return out


def inputs(rows, final, corr=False):
    npool = 8192
    raw = rng.standard_normal((npool, HD))
    import refs as R
    _, codes, sc = R.fp4_quant_rows(raw, 16, True)
    d = dict(rows=rows, Hh=Hh, ring=not final, has_comp=True, packed=True, final=final, has_base=not final, n_idx=512,
             q=C.rand_bf16(rng, (rows, Hh, HD), 0.3), sink=np.zeros(Hh, np.float32), scale=float(HD ** -0.5),
             codes=R.fp4_pack(codes), scales=sc,
             idx=correlated_picks(rows, npool) if corr else rng.integers(0, npool, (rows, 512)).astype(np.int64))
    if final:
        d["pos"] = np.arange(rows, dtype=np.int64) + 1000
        d["wsrc"] = C.rand_bf16(rng, (rows, HD))
        d["wlo"] = np.array([1000], np.int64)
        d["ring_size"] = 16
        d["rbase"] = d["cbase"] = None
    else:
        d["pos"] = rng.integers(200, 3000, rows).astype(np.int64)
        d["wsrc"] = C.rand_bf16(rng, (576, HD))
        d["wlo"] = np.zeros(1, np.int64)
        d["ring_size"] = 144
        d["rbase"] = d["cbase"] = np.zeros(rows, np.int64)
    return d


def triton_args(d, out, parts):
    pos = Buf.of(d["pos"])
    base = Buf.of(d["rbase"]) if d["rbase"] is not None else pos
    pt = "*bf16" if d["final"] else "*fp32"
    pm, pl, po = (out, out, out) if d["final"] else parts
    return dict(Q=P(Buf.of(d["q"]), "*bf16"), WSRC=P(Buf.of(d["wsrc"]), "*bf16"), WLO=P(Buf.of(d["wlo"]), "*i64"),
                COMP=P(Buf.of(d["codes"]), "*u8"), CSC=P(Buf.of(d["scales"]), "*u8"), IDX=P(Buf.of(d["idx"]), "*i64"),
                POS=P(pos, "*i64"), PM=P(pm, pt), PL=P(pl, pt), PO=P(po, pt), SINK=P(Buf.of(d["sink"]), "*fp32"),
                OUT=P(out, "*bf16"), scale=d["scale"], RBASE=P(base, "*i64"), CBASE=P(base, "*i64"),
                ring_size=d["ring_size"], n_idx=512)


def main():
    keep = []
    kdec = T.TA.consts_of(next(c for c in L["_sparse_attn_part"] if c["consts"]["NBLK"] == 20 and not c["consts"]["FINAL"]))
    kfin = T.TA.consts_of(next(c for c in L["_sparse_attn_part"] if c["consts"]["FINAL"] and c["consts"]["NBLK"] == 20))
    C.gpu_lock()
    cases = [(rows, final, inputs(rows, final)) for rows, final in
             ((1, False), (2, False), (4, False), (6, False), (16, False), (512, True), (2048, True))]
    C.record(kernel="window", case="quiet", ok=None, load=C.quiet_window("w-tri-attn"))
    for rows, final, d in cases:
        reps = 3 if final else 20
        C.record(kernel="case", case=f"rows={rows} final={final} hip")
        for _ in range(reps):
            T.hip_attn(d, 1 if final else 8, keep)
            keep.clear()
        C.record(kernel="case", case=f"rows={rows} final={final} triton")
        out = Buf(rows * Hh * HD * 2)
        parts = (Buf(rows * Hh * 8 * 4), Buf(rows * Hh * 8 * 4), Buf(rows * Hh * 8 * HD * 4))
        a = triton_args(d, out, parts)
        for _ in range(reps):
            C.KIT.run("_sparse_attn_part", (rows, 2, 1 if final else 8), a, kfin if final else kdec)
        H.sync()
    C.done()


if __name__ == "__main__":
    main()
