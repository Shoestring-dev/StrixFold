"""The HIP prompt indexer (aux_index.hip, tf_ds_aux_index_rows_h32_d128) against the float64 reference with t_index's
contract (scores within their fp32 bound, -inf masks exact, key index bits exact, tile maxima equal to the stored
keys' maxima), row invariance (a row alone vs in its group of 8 and chunk), split invariance (the key tiles over 1, 7
or the automatic number of workgroups) and agreement with the Triton _index_score_rows on the same inputs
(reported)."""

import ctypes
import json
import os
import sys

import numpy as np

import common as C
import hiprt as H
import t_index as TI
from hiprt import Buf, P

L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
C.log_open("t_hipindex")
OBJ = os.environ.get("TRI_INDEX_HSACO", str(C.ROOT / "build" / "aux_index.hsaco"))
FN = H.Module(open(OBJ, "rb").read()).function("tf_ds_aux_index_rows_h32_d128")
rng = np.random.default_rng(404)
IH, ID = 32, 128
RT = 8                                                  # rows per workgroup


class IndexArgs(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in "q k ks w vis out tmax".split()] + \
               [(n, ctypes.c_int32) for n in "rows n nt flags".split()]


assert ctypes.sizeof(IndexArgs) == 72


def splits_for(rows, nt, target=240):
    """The key-tile split count (grid y): about `target` workgroups in all, at most one tile each."""
    return max(1, min(nt, target // ((rows + RT - 1) // RT)))


def hip_index(q, codes, scales, w, vis, n, keys, tmax_on, splits=None):
    rows = q.shape[0]
    nt = (n + 63) // 64
    splits = splits_for(rows, nt) if splits is None else splits
    out, tm = Buf(rows * n * 8), Buf(rows * nt * 8)
    out.fill(0x5A)
    tm.fill(0x5A)
    b = [Buf.of(q), Buf.of(codes), Buf.of(scales), Buf.of(w), Buf.of(vis)]
    a = IndexArgs(q=b[0].ptr, k=b[1].ptr, ks=b[2].ptr, w=b[3].ptr, vis=b[4].ptr, out=out.ptr, tmax=tm.ptr, rows=rows,
                  n=n, nt=nt, flags=(1 if keys else 0) | (2 if tmax_on else 0))
    H.launch(FN, ((rows + RT - 1) // RT, splits, 1), (256, 1, 1), 0, [a])
    H.sync()
    raw = out.download((rows, n), np.int64 if keys else np.float32)
    return raw, (tm.download((rows, nt), np.int64) if keys and tmax_on else None), (a, b, out, tm)


def main():
    kr = TI.consts_of(next(c for c in L["_index_score_rows"] if c["consts"]["KEYS"] and c["consts"]["HAS_TMAX"]
                           and "nt" not in c["consts"]))
    for rows, n, keys, tmax_on in ((1, 17, False, False), (17, 8, True, True), (5, 3072, True, True), (64, 16384, True, True),
                                   (130, 4096, False, False), (256, 6144, True, True), (33, 70, True, True)):
        codes, scales, deq = TI.make_pool(n)
        q = C.rand_bf16(rng, (rows, IH, ID), 0.5)
        w = C.rand_bf16(rng, (rows, IH), 0.2)
        vis = np.sort(rng.integers(1, n + 1, rows)).astype(np.int64)
        vis[rng.random(rows) < 0.1] = 0                     # rows that see nothing
        raw, tm, _ = hip_index(q, codes, scales, w, vis, n, keys, tmax_on)
        ref, bnd = TI.score_ref(q, w, deq, np.zeros(rows, np.int64), vis, n)
        tag = f"rows={rows} n={n} keys={keys} tmax={tmax_on}"
        if keys:
            got_s, got_i = TI.scores_of_keys(raw)
            TI.check_scores("aux.index_rows", tag, got_s, ref, bnd)
            idx_ok = bool(np.all(got_i == np.arange(n)[None, :]))
            tm_ok = True
            if tmax_on:
                nt = (n + 63) // 64
                pad = np.full((rows, nt * 64), np.iinfo(np.int64).min, np.int64)
                pad[:, :n] = raw
                tm_ok = bool(np.array_equal(tm, pad.reshape(rows, nt, 64).max(-1)))
            C.record(kernel="aux.index_rows", case=tag + " key index bits / tile maxima", ok=idx_ok and tm_ok,
                     index_exact=idx_ok, tmax_exact=tm_ok)
        else:
            TI.check_scores("aux.index_rows", tag, raw.astype(np.float64), ref, bnd)
        # split invariance: one workgroup for all the tiles, and an uneven split, give the same bits
        same = True
        for sp in (1, 7):
            other, _, _ = hip_index(q, codes, scales, w, vis, n, keys, tmax_on, splits=sp)
            same &= bool(np.array_equal(other, raw))
        C.record(kernel="aux.index_rows", case=tag + " split invariance (1, 7, auto)", ok=same)
        # row invariance: rows alone (each in a group of its own) give the same bits
        same = True
        for r in sorted({0, rows // 2, rows - 1}):
            one, _, _ = hip_index(q[r:r + 1], codes, scales, w[r:r + 1], vis[r:r + 1], n, keys, tmax_on)
            same &= bool(np.array_equal(one[0], raw[r]))
        C.record(kernel="aux.index_rows", case=tag + " row invariance", ok=same)
        # the Triton kernel on the same inputs (keys variants only): agreement
        if keys and tmax_on and n % 16 == 0:                 # (the recorded Triton variants hold 16-divisible n)
            nt = (n + 63) // 64
            out, tmb = Buf(rows * n * 8), Buf(rows * nt * 8)
            C.KIT.run("_index_score_rows", (rows, nt, 1), dict(
                Q=P(Buf.of(q), "*bf16"), K=P(Buf.of(codes), "*u8"), KS=P(Buf.of(scales), "*u8"), Wt=P(Buf.of(w), "*bf16"),
                VIS=P(Buf.of(vis), "*i64"), OUT=P(out, "*i64"), n=n, rows=rows, TMAX=P(tmb, "*i64"), nt=nt), kr)
            tri = out.download((rows, n), np.int64)
            a_s, _ = TI.scores_of_keys(raw)
            b_s, _ = TI.scores_of_keys(tri)
            fin = np.isfinite(b_s)
            C.record(kernel="aux.index_rows", case=tag + " vs Triton", ok=None, bits_equal=float(np.mean(raw == tri)),
                     max_abs_diff=float(np.max(np.abs(a_s[fin] - b_s[fin]))) if fin.any() else 0.0)
    sys.exit(C.done())


if __name__ == "__main__":
    main()
