"""Device time per launch of the auxiliary kernels at the model's shapes: decode (1 row), verify windows (2-6 rows)
and prompt chunks (512-8192 rows), each with the variant the engine's wrapper picks for that row count (the recorded
variants; a prompt row count that is 16-divisible falls back as aot.zig does). Inputs are random of the right
shapes; the indexer and attention sizes follow a position (context) given per row. Results: results/t_perf.jsonl."""

import json
import sys

import numpy as np

import common as C
import hiprt as H
import refs as R
from hiprt import Buf, P

L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
rng = np.random.default_rng(5)
C.log_open("t_perf")
D, HC, NB, WIDE = 5120, 4, 40, 20480
ITERS = 30


def k_of(fn, i):
    c = L[fn][i]
    return {k: (v if not isinstance(v, dict) else float(v["float"])) for k, v in c["consts"].items()}


def pick(fn, pred):
    for i, c in enumerate(L[fn]):
        if pred(c, k_of(fn, i)):
            return k_of(fn, i)
    raise KeyError(fn)


def timeit(name, case, call, rows, nbytes=None):
    call()
    H.sync()
    ms = H.time_ms(call, iters=ITERS)
    rec = dict(kernel=name, case=case, rows=rows, us=round(ms * 1000, 2))
    if nbytes:
        rec["GBps"] = round(nbytes / (ms * 1e-3) / 1e9, 1)
    C.record(**rec)


def bf(shape, s=1.0):
    return Buf.of(C.rand_bf16(rng, shape, s))


def perf_norms():
    for D_ in (5120, 1280, 512, 128):
        for rows in (1, 6, 2048, 8192):
            k = pick("_rmsnorm", lambda c, k: k["D"] == D_)
            x, w, o = bf((rows, D_)), bf((D_,)), Buf(rows * D_ * 2)
            args = dict(X=P(x, "*bf16"), xs=D_, W=P(w, "*bf16"), OUT=P(o, "*bf16"), os_=D_, eps=1e-20)
            timeit("_rmsnorm", f"D={D_}", lambda: C.KIT.run("_rmsnorm", (rows, 1, 1), args, k), rows, rows * D_ * 4)
    k = k_of("_kv_norm_rope", 0)
    cos = Buf.of(np.zeros((70000, 32), np.float32))
    for rows in (1, 6, 512, 2048, 8192):
        y, w, out = bf((rows, 512)), bf((512,)), Buf(rows * 1024)
        ring = Buf(max(rows, 144) * 1024)
        slots = Buf.of(np.arange(rows, dtype=np.int64) % 144)
        pos = Buf.of(rng.integers(0, 60000, rows).astype(np.int64))
        args = dict(Y=P(y, "*bf16"), W=P(w, "*bf16"), COS=P(cos, "*fp32"), SIN=P(cos, "*fp32"), POS=P(pos, "*i64"),
                    OUT=P(out, "*bf16"), RING=P(ring, "*bf16"), SLOT_OF=P(slots, "*i64"), ring_size=144, eps=1e-20)
        timeit("_kv_norm_rope", "D=512", lambda: C.KIT.run("_kv_norm_rope", (rows, 1, 1), args, k), rows)
    for HD in (512, 128):
        k = pick("_rope_heads", lambda c, k: k["H"] == 32 and k["HD"] == HD and not k["INV"])
        for rows in (1, 6, 2048, 8192):
            x = bf((rows, 32, HD))
            pos = Buf.of(rng.integers(0, 60000, rows).astype(np.int64))
            args = dict(X=P(x, "*bf16"), COS=P(cos, "*fp32"), SIN=P(cos, "*fp32"), POS=P(pos, "*i64"))
            timeit("_rope_heads", f"H=32 HD={HD}", lambda: C.KIT.run("_rope_heads", (rows, 2, 1), args, k), rows)


def perf_hc():
    fn = Buf.of((rng.standard_normal((24, WIDE)) * 0.01).astype(np.float32))
    sc, base, nw = Buf.of(np.ones(3, np.float32)), Buf.of(np.zeros(24, np.float32)), bf((D,))
    hc_rb = {1: 1, 2: 2, 3: 3, 4: 2, 5: 3, 6: 2, 7: 4, 8: 4}
    for rows in (1, 2, 4, 6, 16):
        rb = hc_rb.get(rows, 4 if rows <= 12 else 8)
        x, xo = bf((rows, HC, D)), Buf(rows * HC * D * 2)
        g = Buf.of((rng.standard_normal((2, rows, D))).astype(np.float32))
        post, comb = Buf.of(np.ones((rows, 4), np.float32)), Buf.of(np.ones((rows, 16), np.float32) / 4)
        part, out = Buf(rows * NB * 32 * 4), Buf(rows * D * 2)
        pre = Buf.of(np.ones((rows, 4), np.float32))
        # the served decode-window mHC step (hc_dots): _hc_post_only, _hc_finish_s (SINK 0), then on the side
        # stream _hc_mix_part (not posted) and _hc_sinkhorn
        kpo = k_of("_hc_post_only", 0)
        a1 = dict(X=P(x, "*bf16"), XO=P(xo, "*bf16"), G=P(g, "*fp32"), RS=rows * D, POST=P(post, "*fp32"),
                  COMB=P(comb, "*fp32"))
        timeit("_hc_post_only", "posted streams", lambda: C.KIT.run("_hc_post_only", (rows, NB, 1), a1, kpo), rows)
        kfs = pick("_hc_finish_s", lambda c, k: not k["SINK"])
        a2 = dict(X=P(xo, "*bf16"), PART=P(part, "*fp32"), BASE=P(base, "*fp32"), SCALE=P(sc, "*fp32"),
                  PRE_IN=P(pre, "*fp32"), NW=P(nw, "*bf16"), OUT=P(out, "*bf16"), PRE_OUT=P(pre, "*fp32"),
                  POST=P(post, "*fp32"), COMB=P(comb, "*fp32"), eps=1e-20, hc_eps=1e-6)
        for i in range(4):
            a2[f"S{i}"] = a2[f"H{i}"] = P(out, "*bf16")
        timeit("_hc_finish_s", "collapse+norm (deferred)", lambda: C.KIT.run("_hc_finish_s", (rows, 1, 1), a2, kfs), rows)
        cands = [k_of("_hc_mix_part", i) for i, c in enumerate(L["_hc_mix_part"]) if not c["consts"]["POSTED"]
                 and c["consts"]["RB"] == rb and not c["consts"]["PDL"]]
        if cands:
            a3 = dict(X=P(xo, "*bf16"), XO=P(xo, "*bf16"), G=P(xo, "*bf16"), RS=0, POST=P(post, "*fp32"),
                      COMB=P(comb, "*fp32"), FN=P(fn, "*fp32"), PART=P(part, "*fp32"), rows=rows)
            kk = dict(cands[0])
            kk.pop("rows", None)
            timeit("_hc_mix_part", f"mixes RB={rb}", lambda: C.KIT.run("_hc_mix_part", ((rows + rb - 1) // rb, NB, 3), a3, kk), rows)
        ksk = k_of("_hc_sinkhorn", 0)
        a4 = {n: a2[n] for n in ("PART", "BASE", "SCALE", "PRE_OUT", "POST", "COMB", "eps", "hc_eps")}
        timeit("_hc_sinkhorn", "", lambda: C.KIT.run("_hc_sinkhorn", (rows, 1, 1), a4, ksk), rows)
    # prompt chunks: _hc_mix_pf (posted) + _hc_finish
    kpf = k_of("_hc_mix_pf", 1)
    kfin = k_of("_hc_finish", 0)
    for rows in (512, 2048, 8192):
        x, xo = bf((rows, HC, D)), Buf(rows * HC * D * 2)
        g = Buf.of((rng.standard_normal((2, rows, D))).astype(np.float32))
        post, comb = Buf.of(np.ones((rows, 4), np.float32)), Buf.of(np.ones((rows, 16), np.float32) / 4)
        part, out, pre = Buf(rows * NB * 32 * 4), Buf(rows * D * 2), Buf.of(np.ones((rows, 4), np.float32))
        a = dict(X=P(x, "*bf16"), XO=P(xo, "*bf16"), G=P(g, "*fp32"), RS=rows * D, POST=P(post, "*fp32"),
                 COMB=P(comb, "*fp32"), FN=P(fn, "*fp32"), PART=P(part, "*fp32"), rows=rows)
        nbytes = rows * HC * D * 2 * 2 + rows * D * 8
        timeit("_hc_mix_pf", "posted + mixes RB=8", lambda: C.KIT.run("_hc_mix_pf", (((rows + 7) // 8) * NB, 1, 1), a, kpf), rows, nbytes)
        a2 = dict(X=P(xo, "*bf16"), PART=P(part, "*fp32"), BASE=P(base, "*fp32"), SCALE=P(sc, "*fp32"),
                  PRE_IN=P(pre, "*fp32"), NW=P(nw, "*bf16"), OUT=P(out, "*bf16"), PRE_OUT=P(pre, "*fp32"),
                  POST=P(post, "*fp32"), COMB=P(comb, "*fp32"), eps=1e-20, hc_eps=1e-6)
        timeit("_hc_finish", "sinkhorn + collapse + norm", lambda: C.KIT.run("_hc_finish", (rows, 1, 1), a2, kfin), rows,
               rows * HC * D * 2 + rows * D * 2)


def perf_route():
    for rows in (1, 2, 4, 6, 16):
        kp = pick("_rowmm_parts", lambda c, k: k["N"] == 384 and k["RB"] in (rows,)) if rows != 5 else None
        if kp is None:
            continue
        x, w, s = bf((rows, 5120)), Buf.of((rng.standard_normal((384, 5120)) * 0.02).astype(np.float16)), Buf(rows * 20 * 384 * 4)
        bn = kp["BN"]
        a = dict(X=P(x, "*bf16"), xs=5120, W=P(w, "*fp16"), S=P(s, "*fp32"), rows=rows)
        kk = dict(kp)
        kk.pop("rows", None)
        timeit("_rowmm_parts", "MoE gate N=384", lambda: C.KIT.run("_rowmm_parts", ((384 + bn - 1) // bn, 20, 1), a, kk), rows, 384 * 5120 * 2)
        kr = pick("_route_parts", lambda c, k: k["NE"] == 384)
        pk, wt = Buf(rows * 7 * 4), Buf(rows * 7 * 4)
        a2 = dict(S=P(s, "*fp32"), BIAS=P(Buf.of(np.zeros(384, np.float32)), "*fp32"), PICK=P(pk, "*i32"), WTS=P(wt, "*fp32"),
                  scale=1.5, shared_id=384)
        timeit("_route_parts", "top-6 of 384", lambda: C.KIT.run("_route_parts", (rows, 1, 1), a2, kr), rows)
        kw = pick("_rowmm2", lambda c, k: k["WTS"])
        w2, o2 = Buf.of((rng.standard_normal((32, 5120)) * 0.02).astype(np.float16)), Buf(rows * 64)
        a3 = dict(X=P(x, "*bf16"), xs=5120, W=P(w2, "*fp16"), OUT=P(o2, "*bf16"), scale=0.015625)
        timeit("_rowmm2", "indexer weights (WTS) N=32", lambda: C.KIT.run("_rowmm2", (rows, 32, 1), a3, kw), rows, 32 * 5120 * 2)
    kr = pick("_route", lambda c, k: k["NE"] == 384)
    for rows in (512, 2048, 8192):
        s = Buf.of((rng.standard_normal((rows, 384))).astype(np.float32))
        pk, wt = Buf(rows * 28), Buf(rows * 28)
        a = dict(L=P(s, "*fp32"), BIAS=P(Buf.of(np.zeros(384, np.float32)), "*fp32"), PICK=P(pk, "*i32"), WTS=P(wt, "*fp32"),
                 scale=1.5, shared_id=384)
        timeit("_route", "prompt top-6 of 384", lambda: C.KIT.run("_route", (rows, 1, 1), a, kr), rows)


def perf_index():
    IH, ID = 32, 128
    for rows, n in ((1, 4096), (1, 32768), (6, 32768), (1, 131072), (6, 131072)):
        k = pick("_index_score", lambda c, k: k["KEYS"] and k["HAS_TMAX"] and not k["HAS_BASE"] and not k["DEAD_PAD"]
                 and "n" not in k)
        codes, scales = Buf(n * 64), Buf(n * 4)
        codes.fill(0x22)
        scales.fill(127)
        q, w = bf((rows, IH, ID), 0.5), bf((rows, IH), 0.2)
        vis = Buf.of(np.full(rows, n, np.int64))
        nt = n // 64
        out, tm = Buf(rows * n * 8), Buf(rows * nt * 8)
        a = dict(Q=P(q, "*bf16"), K=P(codes, "*u8"), KS=P(scales, "*u8"), Wt=P(w, "*bf16"), VIS=P(vis, "*i64"),
                 OUT=P(out, "*i64"), n=n, KB=P(vis, "*i64"), CAND=P(vis, "*i64"), cs=0, TMAX=P(tm, "*i64"), nt=nt)
        timeit("_index_score", f"decode keys+tmax n={n}", lambda: C.KIT.run("_index_score", (rows, nt, 1), a, k), rows, n * 68)
    kr = pick("_index_score_rows", lambda c, k: k["KEYS"] and k["HAS_TMAX"] and "nt" not in k)
    for rows, n in ((512, 8192), (2048, 8192), (2048, 32768), (2048, 65536)):
        codes, scales = Buf(n * 64), Buf(n * 4)
        codes.fill(0x22)
        scales.fill(127)
        q, w = bf((rows, IH, ID), 0.5), bf((rows, IH), 0.2)
        vis = Buf.of(np.full(rows, n, np.int64))
        nt = n // 64
        out, tm = Buf(rows * n * 8), Buf(rows * nt * 8)
        a = dict(Q=P(q, "*bf16"), K=P(codes, "*u8"), KS=P(scales, "*u8"), Wt=P(w, "*bf16"), VIS=P(vis, "*i64"),
                 OUT=P(out, "*i64"), n=n, rows=rows, TMAX=P(tm, "*i64"), nt=nt)
        flops = 2 * rows * n * IH * ID
        call = lambda: C.KIT.run("_index_score_rows", (rows, nt, 1), a, kr)
        call()
        H.sync()
        ms = H.time_ms(call, iters=5, warmup=1)
        C.record(kernel="_index_score_rows", case=f"prompt keys n={n}", rows=rows, us=round(ms * 1000, 1),
                 TFLOPS=round(flops / (ms * 1e-3) / 1e12, 2))
    for kk_n in (1024, 2048, 4096):
        k = pick("_topk_sel", lambda c, k: k["N"] == kk_n and k["KK"] == 512)
        for rows in (1, 6, 2048):
            keys = Buf.of(rng.integers(-2**62, 2**62, (rows, kk_n)).astype(np.int64))
            vis, out = Buf.of(np.full(rows, kk_n, np.int64)), Buf(rows * 512 * 8)
            a = dict(KEYS=P(keys, "*i64"), VIS=P(vis, "*i64"), OUT=P(out, "*i64"))
            timeit("_topk_sel", f"N={kk_n} K=512", lambda: C.KIT.run("_topk_sel", (rows, 1, 1), a, k), rows)


def perf_attn():
    HD, Hh = 512, 32
    pm = lambda r: (Buf(r * Hh * 8 * 4), Buf(r * Hh * 8 * 4), Buf(r * Hh * 8 * HD * 4))
    kpart = pick("_sparse_attn_part", lambda c, k: k["RING"] and k["HAS_COMP"] and k["PACKED"] and k["HAS_BASE"]
                 and not k["FINAL"] and k["NBLK"] == 20)
    kmerge = k_of("_sparse_attn_merge_rot2", 1)
    for rows in (1, 2, 4, 6, 16):
        q, out = bf((rows, Hh, HD), 0.3), Buf(rows * Hh * HD * 2)
        wsrc = bf((576, HD))
        npool = 4096
        codes, scales = Buf(npool * 256), Buf(npool * 32)
        codes.fill(0x22)
        scales.fill(0x38)
        idx = Buf.of(rng.integers(0, npool, (rows, 512)).astype(np.int64))
        pos = Buf.of(rng.integers(200, 3000, rows).astype(np.int64))
        base = Buf.of(np.zeros(rows, np.int64))
        p0, p1, p2 = pm(rows)
        a = dict(Q=P(q, "*bf16"), WSRC=P(wsrc, "*bf16"), WLO=P(base, "*i64"), COMP=P(codes, "*u8"), CSC=P(scales, "*u8"),
                 IDX=P(idx, "*i64"), POS=P(pos, "*i64"), PM=P(p0, "*fp32"), PL=P(p1, "*fp32"), PO=P(p2, "*fp32"),
                 SINK=P(Buf.of(np.zeros(Hh, np.float32)), "*fp32"), OUT=P(out, "*bf16"), scale=0.0441941738,
                 RBASE=P(base, "*i64"), CBASE=P(base, "*i64"), ring_size=144, n_idx=512)
        timeit("_sparse_attn_part", "decode: window 128 + 512 FP4 picks, 8 splits",
               lambda: C.KIT.run("_sparse_attn_part", (rows, 2, 8), a, kpart), rows)
        if rows == 16 or rows == 1:
            km = kmerge if rows == 16 else k_of("_sparse_attn_merge_rot2", 0)
            cos = Buf.of(np.zeros((70000, 32), np.float32))
            a2 = dict(PM=P(p0, "*fp32"), PL=P(p1, "*fp32"), PO=P(p2, "*fp32"), SINK=a["SINK"], OUT=P(out, "*bf16"),
                      COS=P(cos, "*fp32"), SIN=P(cos, "*fp32"), POS=P(pos, "*i64"),
                      SUH=P(Buf.of(np.ones(Hh * HD, np.float16)), "*fp16"), XH=P(Buf(rows * Hh * HD * 2), "*fp16"), rows=rows)
            timeit("_sparse_attn_merge_rot2", "merge + inverse rope + wo_a rotation",
                   lambda: C.KIT.run("_sparse_attn_merge_rot2", (rows, Hh, 1), a2, km), rows)
    kfin = pick("_sparse_attn_part", lambda c, k: k["FINAL"] and k["PACKED"] and k["NBLK"] == 20)
    for rows in (512, 2048):
        q, out = bf((rows, Hh, HD), 0.3), Buf(rows * Hh * HD * 2)
        wsrc = bf((rows, HD))
        npool = 8192
        codes, scales = Buf(npool * 256), Buf(npool * 32)
        codes.fill(0x22)
        scales.fill(0x38)
        idx = Buf.of(rng.integers(0, npool, (rows, 512)).astype(np.int64))
        pos = Buf.of(np.arange(rows, dtype=np.int64) + 1000)
        wlo = Buf.of(np.array([1000], np.int64))
        a = dict(Q=P(q, "*bf16"), WSRC=P(wsrc, "*bf16"), WLO=P(wlo, "*i64"), COMP=P(codes, "*u8"), CSC=P(scales, "*u8"),
                 IDX=P(idx, "*i64"), POS=P(pos, "*i64"), PM=P(out, "*bf16"), PL=P(out, "*bf16"), PO=P(out, "*bf16"),
                 SINK=P(Buf.of(np.zeros(Hh, np.float32)), "*fp32"), OUT=P(out, "*bf16"), scale=0.0441941738,
                 RBASE=P(pos, "*i64"), CBASE=P(pos, "*i64"), ring_size=16, n_idx=512)
        flops = 2 * 2 * rows * Hh * HD * (128 + 512)
        call = lambda: C.KIT.run("_sparse_attn_part", (rows, 2, 1), a, kfin)
        call()
        H.sync()
        ms = H.time_ms(call, iters=5, warmup=1)
        C.record(kernel="_sparse_attn_part", case="prompt: window 128 + 512 FP4 picks, one pass", rows=rows,
                 us=round(ms * 1000, 1), TFLOPS=round(flops / (ms * 1e-3) / 1e12, 2))


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for t in (perf_norms, perf_hc, perf_route, perf_index, perf_attn):
        if only and t.__name__ not in only:
            continue
        t()
    H.sync()
    C.done()
