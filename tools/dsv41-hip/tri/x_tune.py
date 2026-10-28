"""Tuning experiment: num_warps / num_stages of the heavy kernels (decode attention part, prompt attention, prompt
indexer) -- device time and register use per setting; correctness of a retuned setting is checked by the suite after
the kit is rebuilt with it."""
import copy, json, sys
import numpy as np
import common as C, hiprt as H
from hiprt import Buf, P
L = json.load(open(C.ROOT / "upstream" / "launches.json"))
rng = np.random.default_rng(3)
C.log_open("x_tune")
HD, Hh = 512, 32


def bf(shape, s=1.0):
    return Buf.of(C.rand_bf16(rng, shape, s))


def attn_args(rows, final):
    q, out = bf((rows, Hh, HD), 0.3), Buf(rows * Hh * HD * 2)
    npool = 8192
    codes, scales = Buf(npool * 256), Buf(npool * 32)
    codes.fill(0x22); scales.fill(0x38)
    idx = Buf.of(rng.integers(0, npool, (rows, 512)).astype(np.int64))
    if final:
        wsrc = bf((rows, HD)); pos = Buf.of(np.arange(rows, dtype=np.int64) + 1000); wlo = Buf.of(np.array([1000], np.int64))
        parts = (P(out, "*bf16"),) * 3
        base = pos
    else:
        wsrc = bf((576, HD)); pos = Buf.of(rng.integers(200, 3000, rows).astype(np.int64)); wlo = Buf.of(np.zeros(1, np.int64))
        parts = (P(Buf(rows * Hh * 8 * 4), "*fp32"), P(Buf(rows * Hh * 8 * 4), "*fp32"), P(Buf(rows * Hh * 8 * HD * 4), "*fp32"))
        base = Buf.of(np.zeros(rows, np.int64))
    return dict(Q=P(q, "*bf16"), WSRC=P(wsrc, "*bf16"), WLO=P(wlo, "*i64"), COMP=P(codes, "*u8"), CSC=P(scales, "*u8"),
                IDX=P(idx, "*i64"), POS=P(pos, "*i64"), PM=parts[0], PL=parts[1], PO=parts[2],
                SINK=P(Buf.of(np.zeros(Hh, np.float32)), "*fp32"), OUT=P(out, "*bf16"), scale=0.0441941738,
                RBASE=P(base, "*i64"), CBASE=P(base, "*i64"), ring_size=16 if final else 144, n_idx=512)


def variant(fn, case, warps, stages=None):
    c = copy.deepcopy(case); c["num_warps"] = warps
    import aot_build
    old = dict(aot_build.EXTRA_OPTIONS.get(fn, {}))
    o = dict(old); o.pop("num_warps", None)
    if stages is not None:
        o["num_stages"] = stages
    aot_build.EXTRA_OPTIONS[fn] = o
    try:
        C._EXTRA.clear()
        return C.extra_variant(fn, c)
    finally:
        aot_build.EXTRA_OPTIONS[fn] = old


def main():
    dec = next(c for c in L["_sparse_attn_part"] if c["consts"]["NBLK"] == 20 and not c["consts"]["FINAL"])
    fin = next(c for c in L["_sparse_attn_part"] if c["consts"]["FINAL"] and c["consts"]["NBLK"] == 20)
    for name, case, grid_rows, final in (("decode r=1", dec, 1, False), ("decode r=6", dec, 6, False), ("decode r=16", dec, 16, False),
                                         ("prompt r=2048", fin, 2048, True)):
        a = attn_args(grid_rows, final)
        for warps in (4, 8):
            for stages in (1, 2):
                kv = variant("_sparse_attn_part", case, warps, stages)
                sp = 1 if final else 8
                call = lambda: C.run_variant(kv, (grid_rows, 2, sp), a)
                call(); H.sync()
                ms = H.time_ms(call, iters=10 if final else 50, warmup=2)
                C.record(kernel="_sparse_attn_part", case=f"{name} warps={warps} stages={stages}", us=round(ms * 1000, 1),
                         vgpr=kv["isa_stats"].get("vgpr_count"), spill=kv["isa_stats"].get("vgpr_spill_count"), lds=kv["shared"])
    # prompt indexer warps
    isr = L["_index_score_rows"][5]
    rows, n = 2048, 16384
    codes, scales = Buf(n * 64), Buf(n * 4); codes.fill(0x22); scales.fill(127)
    q, w = bf((rows, 32, 128), 0.5), bf((rows, 32), 0.2)
    vis = Buf.of(np.full(rows, n, np.int64)); nt = n // 64
    out, tm = Buf(rows * n * 8), Buf(rows * nt * 8)
    a = dict(Q=P(q, "*bf16"), K=P(codes, "*u8"), KS=P(scales, "*u8"), Wt=P(w, "*bf16"), VIS=P(vis, "*i64"), OUT=P(out, "*i64"),
             n=n, rows=rows, TMAX=P(tm, "*i64"), nt=nt)
    for warps in (1, 2, 4):
        kv = variant("_index_score_rows", isr, warps)
        call = lambda: C.run_variant(kv, (rows, nt, 1), a)
        call(); H.sync()
        ms = H.time_ms(call, iters=3, warmup=1)
        C.record(kernel="_index_score_rows", case=f"rows={rows} n={n} warps={warps}", us=round(ms * 1000, 1),
                 TFLOPS=round(2 * rows * n * 32 * 128 / (ms * 1e-3) / 1e12, 2), vgpr=kv["isa_stats"].get("vgpr_count"),
                 spill=kv["isa_stats"].get("vgpr_spill_count"))
    C.done()


if __name__ == "__main__":
    main()
