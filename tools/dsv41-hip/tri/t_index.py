"""The indexer: _index_score (decode windows), _index_score_rows (prompt chunks), _index_score_cand (pool-only keys),
_score_keys, _topk_sel, _topk_finish, _gather_tiles (and the pruned selection they compose), _fp4qd_p2 -- every
recorded variant against refs.py: scores within their fp32 bound, int64 keys / selections / packed bytes exactly; and
every derived one (variants.py: each context length the engine serves) at its own n on its exact variant."""

import json
import sys

import numpy as np

import common as C
import hiprt as H
import refs as R
from hiprt import Buf, P

LAUNCHES = C.launches()
NREC = {fn: len(cs) for fn, cs in json.loads((C.ROOT / "upstream" / "launches.json").read_text()).items()}
rng = np.random.default_rng(99)
C.log_open("t_index")
IH, ID, BN = 32, 128, 64
PAD = np.int64(-9223372036854775808)


def consts_of(case):
    return {k: (v if not isinstance(v, dict) else float(v["float"])) for k, v in case["consts"].items()}


def make_pool(npool):
    """Indexer keys: fp4_act_quant (E8M0, 32-blocks) of random rows, packed as the cache holds them."""
    raw = rng.standard_normal((npool, ID)) * np.exp(rng.uniform(-2, 2, (npool, 1)))
    deq, codes, sc = R.fp4_quant_rows(raw, 32, False)
    return R.fp4_pack(codes), sc, deq


def keys_of_scores(score, t):
    """topk_indices' int64 keys: the fp32 score's bits (-0 as +0) in signed total order above, ~index below."""
    s = np.asarray(score, np.float32) + np.float32(0.0)
    bits = s.view(np.int32).astype(np.int64)
    ordered = np.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    return (ordered << 32) | (0xFFFFFFFF - np.asarray(t, np.int64))


def scores_of_keys(keys):
    ordered = (keys >> 32).astype(np.int32)
    bits = np.where(ordered < 0, ordered ^ 0x7FFFFFFF, ordered).astype(np.int32)
    idx = 0xFFFFFFFF - (keys & 0xFFFFFFFF)
    return bits.view(np.float32).astype(np.float64), idx


def score_ref(q, w, deq_pool, kbase, vis, n, cand=None, cb=8):
    """fp64 scores [R, n] (-inf at t >= vis or outside the pool) and their fp32 error bound."""
    rows = q.shape[0]
    qs = C.from_bf16_bits(q)                                   # [R, IH, ID]
    ws = C.from_bf16_bits(w)                                   # [R, IH]
    out = np.full((rows, n), -np.inf)
    bnd = np.zeros((rows, n))
    for r in range(rows):
        v = min(int(vis[r]), n)
        if v <= 0:
            continue
        kk = deq_pool[kbase[r]:kbase[r] + v]                   # [v, ID]
        d = qs[r] @ kk.T                                       # [IH, v]
        ad = np.abs(qs[r]) @ np.abs(kk).T
        terms = np.maximum(d, 0) * ws[r][:, None]
        out[r, :v] = terms.sum(0)
        bnd[r, :v] = (np.abs(ws[r])[:, None] * C.gamma(ID) * ad).sum(0) + C.gamma(IH + 2) * np.abs(terms).sum(0)
    if cand is not None:
        keep = np.repeat(cand, cb, axis=1)[:, :n].astype(bool)
        out = np.where(keep, out, -np.inf)
    return out, bnd


def check_scores(name, case, got, ref, bnd):
    inf = np.isneginf(ref)
    ok_inf = bool(np.all(np.isneginf(got[inf])))
    m = C.cmp_bound(got[~inf], ref[~inf], bnd[~inf]) if (~inf).any() else dict(ok=True)
    m["masked_exact"] = ok_inf
    m["ok"] = m["ok"] and ok_inf
    C.record(kernel=name, case=case, **m)


def t_index_score():
    for ci, c in enumerate(LAUNCHES["_index_score"]):
        k = consts_of(c)
        derived = ci >= NREC["_index_score"]
        rows = c["tensors"]["Q"][1][0]
        n = c["scalars"].get("n", k.get("n"))
        if derived and n > 16384:
            rows = min(rows, 4)                            # rows change only the grid
        nt = (n + BN - 1) // BN
        has_base, keys, has_tmax = bool(k["HAS_BASE"]), bool(k["KEYS"]), bool(k["HAS_TMAX"])
        has_cand, dead_pad = bool(k["HAS_CAND"]), bool(k["DEAD_PAD"])
        npool = n + 4096 if has_base else n
        codes, scales, deq = make_pool(npool)
        q = C.rand_bf16(rng, (rows, IH, ID), 0.5)
        w = C.rand_bf16(rng, (rows, IH), 0.2)
        vis = rng.integers(1, n + 1, rows).astype(np.int64)
        vis[0] = n
        if rows > 1:
            vis[1] = max(1, n // 3)
        kbase = rng.integers(0, npool - n + 1, rows).astype(np.int64) if has_base else np.zeros(rows, np.int64)
        # the pool's block mask, a row's stride padded to 16 bytes as the engine's mask buffers are
        cw = c["scalars"]["cs"] if derived else ((n + 7) // 8 + 15) // 16 * 16
        cand = (rng.random((rows, cw)) < 0.6).astype(np.uint8) if has_cand else None
        cs = cand.shape[1] if has_cand else 0
        out_dt = np.int64 if keys else np.float32
        out = Buf(rows * n * 8)
        out.fill(0x5A)
        tmax = Buf(rows * nt * 8)
        tmax.fill(0x5A)
        visb = Buf.of(vis)
        args = dict(Q=P(Buf.of(q), "*bf16"), K=P(Buf.of(codes), "*u8"), KS=P(Buf.of(scales), "*u8"),
                    Wt=P(Buf.of(w), "*bf16"), VIS=P(visb, "*i64"), OUT=P(out, "*i64" if keys else "*fp32"), n=n,
                    KB=P(Buf.of(kbase), "*i64") if has_base else P(visb, "*i64"),
                    CAND=P(Buf.of(cand), "*u8") if has_cand else P(visb, "*i64"), cs=cs,
                    TMAX=P(tmax, "*i64") if has_tmax else P(out, "*i64" if keys else "*fp32"), nt=nt)
        tag = (f"{'derived' if derived else 'rec'}{ci} rows={rows} n={n} base={has_base} keys={keys} tmax={has_tmax} "
               f"cand={has_cand} pad={dead_pad}")
        if derived:
            if not C.run_exact("_index_score", (rows, nt, 1), args, k, tag):
                continue
        else:
            C.KIT.run("_index_score", (rows, nt, 1), args, k)
        ref, bnd = score_ref(q, w, deq, kbase, vis, n, cand, 8)
        raw = out.download((rows, n), out_dt)
        if not keys:
            check_scores("_index_score", tag, raw.astype(np.float64), ref, bnd)
            continue
        live = np.ones((rows, n), bool)
        if dead_pad:                                       # tiles wholly at or past vis write no keys
            tiles = np.arange(n) // BN * BN
            live = tiles[None, :] < vis[:, None]
        got_s, got_i = scores_of_keys(raw)
        idx_ok = bool(np.all(got_i[live] == np.broadcast_to(np.arange(n), (rows, n))[live]))
        untouched = bool(np.all(raw[~live] == np.int64(0x5A5A5A5A5A5A5A5A)))
        check_scores("_index_score", tag + " (scores in keys)", np.where(live, got_s, -np.inf),
                     np.where(live, ref, -np.inf), bnd)
        C.record(kernel="_index_score", case=tag + " key index bits / dead tiles", ok=idx_ok and untouched,
                 index_exact=idx_ok, dead_tiles_untouched=untouched)
        if has_tmax:
            tm = tmax.download((rows, nt), np.int64)
            want = np.full((rows, nt), PAD)
            for b in range(nt):
                seg = raw[:, b * BN:min(n, (b + 1) * BN)]
                lv = live[:, b * BN:min(n, (b + 1) * BN)]
                want[:, b] = np.where(lv.any(1), np.where(lv, seg, PAD).max(1), PAD)
            C.record(kernel="_index_score", case=tag + " tile maxima", ok=bool(np.array_equal(tm, want)))


def t_index_score_rows():
    for ci, c in enumerate(LAUNCHES["_index_score_rows"]):
        k = consts_of(c)
        n = c["scalars"].get("n", k.get("n"))
        rows = min(c["scalars"].get("rows", k.get("rows", 1)), 256)
        n = min(n, 16384) if n > 64 else n
        nt = (n + BN - 1) // BN
        keys, has_tmax = bool(k["KEYS"]), bool(k["HAS_TMAX"])
        codes, scales, deq = make_pool(n)
        q = C.rand_bf16(rng, (rows, IH, ID), 0.5)
        w = C.rand_bf16(rng, (rows, IH), 0.2)
        vis = np.sort(rng.integers(1, n + 1, rows)).astype(np.int64)
        out = Buf(rows * n * 8)
        tmax = Buf(rows * nt * 8)
        args = dict(Q=P(Buf.of(q), "*bf16"), K=P(Buf.of(codes), "*u8"), KS=P(Buf.of(scales), "*u8"),
                    Wt=P(Buf.of(w), "*bf16"), VIS=P(Buf.of(vis), "*i64"), OUT=P(out, "*i64" if keys else "*fp32"),
                    n=n, rows=rows, TMAX=P(tmax, "*i64") if has_tmax else P(out, "*i64" if keys else "*fp32"), nt=nt)
        C.KIT.run("_index_score_rows", (rows, nt, 1), args, k)
        ref, bnd = score_ref(q, w, deq, np.zeros(rows, np.int64), vis, n)
        tag = f"rec{ci} rows={rows} n={n} keys={keys} tmax={has_tmax}"
        raw = out.download((rows, n), np.int64 if keys else np.float32)
        if keys:
            got_s, got_i = scores_of_keys(raw)
            check_scores("_index_score_rows", tag, got_s, ref, bnd)
            idx_ok = bool(np.all(got_i == np.arange(n)[None, :]))
            tm_ok = True
            if has_tmax:
                tm = tmax.download((rows, nt), np.int64)
                want = np.stack([raw[:, b * BN:(b + 1) * BN].max(1) for b in range(nt)], 1)
                tm_ok = bool(np.array_equal(tm, want))
                if not tm_ok:
                    bad = np.argwhere(tm != want)[:3]
                    print("tmax mismatch at", bad.tolist(), [(int(tm[i, j]), int(want[i, j])) for i, j in bad])
            C.record(kernel="_index_score_rows", case=tag + " key bits / tile maxima", ok=idx_ok and tm_ok,
                     index_exact=idx_ok, tmax_exact=tm_ok)
        else:
            check_scores("_index_score_rows", tag, raw.astype(np.float64), ref, bnd)
        # the head-sum tree's purpose: a row's scores the same whatever shares its program; and the decode-window
        # kernel (one row a program) against the prompt kernel on the same row: report whether they agree bitwise
        if ci == 3 and keys:
            one = Buf(n * 8)
            args1 = dict(args, Q=P(Buf.of(q[5:6]), "*bf16"), Wt=P(Buf.of(w[5:6]), "*bf16"),
                         VIS=P(Buf.of(vis[5:6]), "*i64"), OUT=P(one, "*i64"), rows=1,
                         TMAX=P(Buf(nt * 8), "*i64"))
            C.KIT.run("_index_score_rows", (1, nt, 1), args1, k)
            C.record(kernel="_index_score_rows", case="row invariance (row 5 alone vs in the chunk)",
                     ok=bool(np.array_equal(one.download((1, n), np.int64)[0], raw[5])))


def t_index_score_cand():
    c = LAUNCHES["_index_score_cand"][0]
    k = consts_of(c)
    cb = k["CB"]
    for rows, n, nblk in ((2, 32768, 512), (16, 8192, 256)):
        npool = n + 1024
        codes, scales, deq = make_pool(npool)
        q = C.rand_bf16(rng, (rows, IH, ID), 0.5)
        w = C.rand_bf16(rng, (rows, IH), 0.2)
        vis = rng.integers(n // 2, n + 1, rows).astype(np.int64)
        kbase = rng.integers(0, npool - n + 1, rows).astype(np.int64)
        cblk = np.stack([np.sort(rng.choice(n // cb, nblk, replace=False)) for _ in range(rows)]).astype(np.int32)
        cblk[:, -7:] = -1
        nout = nblk * cb
        out = Buf(rows * nout * 8)
        C.KIT.run("_index_score_cand", (rows, (nout + BN - 1) // BN, 1), dict(
            Q=P(Buf.of(q), "*bf16"), K=P(Buf.of(codes), "*u8"), KS=P(Buf.of(scales), "*u8"), Wt=P(Buf.of(w), "*bf16"),
            VIS=P(Buf.of(vis), "*i64"), OUT=P(out, "*i64"), n=n, nout=nout, KB=P(Buf.of(kbase), "*i64"),
            CBLK=P(Buf.of(cblk), "*i32"), cbs=nblk), k)
        raw = out.download((rows, nout), np.int64)
        ref, bnd = score_ref(q, w, deq, kbase, vis, n)
        pos = (cblk.astype(np.int64)[:, :, None] * cb + np.arange(cb)).reshape(rows, nout)
        valid = (np.repeat(cblk, cb, axis=1) >= 0) & (pos < n)
        got_s, got_i = scores_of_keys(raw)
        pad_ok = bool(np.all(raw[~valid] == PAD))
        idx_ok = bool(np.all(got_i[valid] == pos[valid]))
        refp = np.where(valid, np.take_along_axis(ref, np.clip(pos, 0, n - 1), 1), -np.inf)
        bndp = np.where(valid, np.take_along_axis(bnd, np.clip(pos, 0, n - 1), 1), 0)
        check_scores("_index_score_cand", f"rows={rows} n={n} pool blocks={nblk}", np.where(valid, got_s, -np.inf),
                     refp, bndp)
        C.record(kernel="_index_score_cand", case=f"rows={rows} n={n} key positions / PAD", ok=pad_ok and idx_ok)


def t_index_score_cand_derived():
    """The derived _index_score_cand variants (variants.py: every context the rounds serve), each at its own n and pool
    width on its exact variant: keys of the pool's blocks against score_ref, PAD past n and for no block."""
    for ci, c in enumerate(LAUNCHES["_index_score_cand"][NREC["_index_score_cand"]:], NREC["_index_score_cand"]):
        k = consts_of(c)
        cb, n, nblk = k["CB"], c["scalars"]["n"], c["scalars"]["cbs"]
        rows = 2
        npool = n + 1024
        codes, scales, deq = make_pool(npool)
        q = C.rand_bf16(rng, (rows, IH, ID), 0.5)
        w = C.rand_bf16(rng, (rows, IH), 0.2)
        vis = rng.integers(n // 2, n + 1, rows).astype(np.int64)
        vis[0] = n
        kbase = rng.integers(0, npool - n + 1, rows).astype(np.int64)
        nb_all = (n + cb - 1) // cb
        cblk = np.stack([np.sort(rng.choice(nb_all, nblk, replace=False)) for _ in range(rows)]).astype(np.int32)
        cblk[:, -7:] = -1
        nout = nblk * cb
        out = Buf(rows * nout * 8)
        tag = f"derived{ci} rows={rows} n={n} pool blocks={nblk}"
        if not C.run_exact("_index_score_cand", (rows, (nout + BN - 1) // BN, 1), dict(
                Q=P(Buf.of(q), "*bf16"), K=P(Buf.of(codes), "*u8"), KS=P(Buf.of(scales), "*u8"),
                Wt=P(Buf.of(w), "*bf16"), VIS=P(Buf.of(vis), "*i64"), OUT=P(out, "*i64"), n=n, nout=nout,
                KB=P(Buf.of(kbase), "*i64"), CBLK=P(Buf.of(cblk), "*i32"), cbs=nblk), k, tag):
            continue
        raw = out.download((rows, nout), np.int64)
        ref, bnd = score_ref(q, w, deq, kbase, vis, n)
        pos = (cblk.astype(np.int64)[:, :, None] * cb + np.arange(cb)).reshape(rows, nout)
        valid = (np.repeat(cblk, cb, axis=1) >= 0) & (pos < n)
        got_s, got_i = scores_of_keys(raw)
        refp = np.where(valid, np.take_along_axis(ref, np.clip(pos, 0, n - 1), 1), -np.inf)
        bndp = np.where(valid, np.take_along_axis(bnd, np.clip(pos, 0, n - 1), 1), 0)
        check_scores("_index_score_cand", tag, np.where(valid, got_s, -np.inf), refp, bndp)
        C.record(kernel="_index_score_cand", case=tag + " key positions / PAD",
                 ok=bool(np.all(raw[~valid] == PAD)) and bool(np.all(got_i[valid] == pos[valid])))


def t_gather_tiles():
    """Every _gather_tiles variant, recorded and derived, at its own n (= its row stride ks) on its exact variant:
    each picked position's 64-key tile, PAD past n and for a position of -1."""
    for ci, c in enumerate(LAUNCHES["_gather_tiles"]):
        k = consts_of(c)
        n, ks, kk, bn = c["scalars"]["n"], c["scalars"]["ks"], k["KK"], k["BN"]
        rows = 2
        keys = rng.integers(-(1 << 62), 1 << 62, (rows, ks), dtype=np.int64)
        tpos = rng.integers(0, n, (rows, kk)).astype(np.int64)
        tpos[:, -5:] = -1
        tpos[0, 0] = n - 1                                 # the last, partial tile
        out = Buf(rows * kk * bn * 8)
        tag = f"{'derived' if ci >= NREC['_gather_tiles'] else 'rec'}{ci} rows={rows} n={n}"
        if not C.run_exact("_gather_tiles", (rows, kk, 1), dict(KEYS=P(Buf.of(keys), "*i64"), TPOS=P(Buf.of(tpos), "*i64"),
                           OUT=P(out, "*i64"), n=n, ks=ks), k, tag):
            continue
        t = (tpos // bn * bn)[:, :, None] + np.arange(bn)
        ok_t = (t < n) & (tpos[:, :, None] >= 0)
        want = np.where(ok_t, np.take_along_axis(keys, np.clip(t, 0, ks - 1).reshape(rows, -1), 1).reshape(t.shape),
                        PAD).reshape(rows, kk * bn)
        C.record(kernel="_gather_tiles", case=tag, ok=bool(np.array_equal(out.download((rows, kk * bn), np.int64), want)),
                 exact=True)


def t_score_keys():
    for ci, c in enumerate(LAUNCHES["_score_keys"]):
        k = consts_of(c)
        rows = c["tensors"]["S"][1][0]
        n = c["scalars"]["n"]
        has_tmax = bool(k["HAS_TMAX"])
        s = (rng.standard_normal((rows, n))).astype(np.float32)
        s[rng.random((rows, n)) < 0.2] = -np.inf
        s[0, :7] = [0.0, -0.0, np.inf, -np.inf, 1e-45, -1e-45, 3.0]
        nt = (n + 63) // 64
        out, tm = Buf(rows * n * 8), Buf(rows * nt * 8)
        args = dict(S=P(Buf.of(s), "*fp32"), OUT=P(out, "*i64"), n=n, TMAX=P(tm, "*i64") if has_tmax else P(out, "*i64"),
                    nt=nt)
        tag = f"{'derived' if ci >= NREC['_score_keys'] else 'rec'}{ci} rows={rows} n={n} tmax={has_tmax}"
        if ci >= NREC["_score_keys"]:
            if not C.run_exact("_score_keys", (rows, (n + 1023) // 1024, 1), args, k, tag):
                continue
        else:
            C.KIT.run("_score_keys", (rows, (n + 1023) // 1024, 1), args, k)
        want = keys_of_scores(s, np.arange(n)[None, :])
        got = out.download((rows, n), np.int64)
        ok = bool(np.array_equal(got, want))
        if has_tmax:                                   # each 64-key tile's largest (a partial last tile: its keys')
            wt = np.stack([want[:, b * 64:min(n, (b + 1) * 64)].max(1) for b in range(nt)], 1)
            ok &= bool(np.array_equal(tm.download((rows, nt), np.int64), wt))
        C.record(kernel="_score_keys", case=tag, ok=ok, exact=True)


def topk_ref(keys, kk, vis):
    top = np.sort(keys, axis=1)[:, -kk:]
    idx = np.sort(0xFFFFFFFF - (top & 0xFFFFFFFF), axis=1)
    return np.where(idx < vis[:, None], idx, -1)


def unique_keys(rows, n):
    s = rng.standard_normal((rows, n)).astype(np.float32)
    s[rng.random((rows, n)) < 0.1] = -np.inf
    s[:, :3] = 0.5                                   # ties: broken by the index
    return keys_of_scores(s, np.arange(n)[None, :])


def t_topk():
    for ci, c in enumerate(LAUNCHES["_topk_sel"]):
        k = consts_of(c)
        n, kk = k["N"], k["KK"]
        for rows in (1, 16, c["tensors"]["KEYS"][1][0]):
            keys = unique_keys(rows, n)
            vis = rng.integers(1, n + 1, rows).astype(np.int64)
            out = Buf(rows * kk * 8)
            args = dict(KEYS=P(Buf.of(keys), "*i64"), VIS=P(Buf.of(vis), "*i64"), OUT=P(out, "*i64"))
            tag = f"{'derived' if ci >= NREC['_topk_sel'] else 'rec'}{ci} N={n} K={kk} rows={rows}"
            if ci >= NREC["_topk_sel"]:
                if not C.run_exact("_topk_sel", (rows, 1, 1), args, k, tag):
                    continue
            else:
                C.KIT.run("_topk_sel", (rows, 1, 1), args, k)
            ok = bool(np.array_equal(out.download((rows, kk), np.int64), topk_ref(keys, kk, vis)))
            C.record(kernel="_topk_sel", case=tag, ok=ok, exact=True)
    for ci, c in enumerate(LAUNCHES["_topk_finish"]):
        k = consts_of(c)
        kk = k["KK"]
        rows = c["tensors"]["TOP"][1][0]
        keys = unique_keys(rows, 4 * kk)
        top = np.stack([rng.permutation(np.sort(keys[r])[-kk:]) for r in range(rows)])
        vis = rng.integers(1, 4 * kk + 1, rows).astype(np.int64)
        out = Buf(rows * kk * 8)
        C.KIT.run("_topk_finish", (rows, 1, 1), dict(TOP=P(Buf.of(top), "*i64"), VIS=P(Buf.of(vis), "*i64"),
                                                    OUT=P(out, "*i64")), k)
        ok = bool(np.array_equal(out.download((rows, kk), np.int64), topk_ref(keys, kk, vis)))
        C.record(kernel="_topk_finish", case=f"rec{ci} K={kk} rows={rows}", ok=ok, exact=True)


def t_pruned():
    """topk_select_pruned's composition (score keys with tile maxima -> top-k tiles -> _gather_tiles -> top-k) equals
    the top-k of every key; and _gather_tiles alone exactly."""
    ks = consts_of(next(c for c in LAUNCHES["_score_keys"] if c["consts"]["HAS_TMAX"]))
    kg = consts_of(LAUNCHES["_gather_tiles"][0])
    sel = {consts_of(c)["N"]: consts_of(c) for c in LAUNCHES["_topk_sel"] if consts_of(c)["KK"] == 512}
    kk = 512
    for rows, n in ((2, 131072), (4, 32768)):
        s = rng.standard_normal((rows, n)).astype(np.float32)
        vis = rng.integers(n // 2, n + 1, rows).astype(np.int64)
        s[np.arange(n)[None, :] >= vis[:, None]] = -np.inf
        nt = n // 64
        keys, tm = Buf(rows * n * 8), Buf(rows * nt * 8)
        C.KIT.run("_score_keys", (rows, n // 1024, 1), dict(S=P(Buf.of(s), "*fp32"), OUT=P(keys, "*i64"), n=n,
                  TMAX=P(tm, "*i64"), nt=nt), ks)
        tmax = tm.download((rows, nt), np.int64)
        every = np.full(rows, 1 << 40, np.int64)
        tpos = Buf(rows * kk * 8)
        if nt not in sel:
            C.record(kernel="_gather_tiles", case=f"n={n}: no recorded _topk_sel for N={nt}", ok=None)
            continue
        C.KIT.run("_topk_sel", (rows, 1, 1), dict(KEYS=P(tm, "*i64"), VIS=P(Buf.of(every), "*i64"),
                                                 OUT=P(tpos, "*i64")), sel[nt])
        cand = Buf(rows * kk * 64 * 8)
        C.KIT.run("_gather_tiles", (rows, kk, 1), dict(KEYS=P(keys, "*i64"), TPOS=P(tpos, "*i64"), OUT=P(cand, "*i64"),
                                                      n=n, ks=n), kg)
        allk = keys.download((rows, n), np.int64)
        tp = tpos.download((rows, kk), np.int64)
        want_cand = np.stack([np.concatenate([allk[r, (p // 64) * 64:(p // 64) * 64 + 64] for p in tp[r]])
                              for r in range(rows)])
        got_cand = cand.download((rows, kk * 64), np.int64)
        C.record(kernel="_gather_tiles", case=f"rows={rows} n={n}", ok=bool(np.array_equal(got_cand, want_cand)))
        out = Buf(rows * kk * 8)
        # the last step's N = k * 64 = 32768 keys a row: past TOPK_FUSED_MAX, so the engine takes the radix top-k
        # (ops.topkI64) + _topk_finish there; check the composition with the numpy top-k in that place
        got_final = topk_ref(got_cand, kk, vis)
        C.record(kernel="topk_select_pruned", case=f"rows={rows} n={n} (pruned == full top-k)",
                 ok=bool(np.array_equal(got_final, topk_ref(allk, kk, vis))))


def t_fp4qd():
    for ci, c in enumerate(LAUNCHES["_fp4qd_p2"]):
        k = consts_of(c)
        rows = c["tensors"]["X"][1][0]
        x = C.rand_bf16(rng, (rows, IH, ID), 1.0)
        xs = C.from_bf16_bits(x).reshape(-1, 32)
        xs[0] = 0
        xs[1, :] = 1e-30
        x = C.bf16_bits(xs.reshape(rows, IH, ID))
        nblk = x.size // 32
        out = Buf(x.size * 2)
        C.KIT.run("_fp4qd_p2", ((nblk + 31) // 32, 1, 1), dict(X=P(Buf.of(x), "*bf16"), OUT=P(out, "*bf16"), nblk=nblk), k)
        deq, _, _ = R.fp4_quant_rows(C.from_bf16_bits(x).reshape(-1, 32), 32, False)
        got = C.from_bf16_bits(out.download(x.shape, np.uint16)).reshape(-1, 32)
        want = R.bf16(deq)
        C.record(kernel="_fp4qd_p2", case=f"rec{ci} rows={rows}", ok=bool(np.array_equal(got, want)),
                 mismatches=int((got != want).sum()))


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for t in (t_index_score, t_index_score_rows, t_index_score_cand, t_index_score_cand_derived, t_score_keys, t_topk,
              t_pruned, t_gather_tiles, t_fp4qd):
        if only and t.__name__ not in only:
            continue
        t()
    H.sync()
    sys.exit(C.done())
