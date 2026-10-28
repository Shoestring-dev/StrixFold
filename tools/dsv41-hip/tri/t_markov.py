"""DSpark's Markov steps: _bias (a token's bias row fp16(markov_head @ fp16(markov_embed[t]))), _score (logit + bias,
the best (value, index) of each tile in torch.argmax's order), _finish (a row's best, or its (value, index) for the
ranks' gather), _pick (the best over the ranks) -- every recorded variant against numpy; a whole TP2 step (both
vocabulary halves on this GPU) against the full-vocabulary argmax; cached rows giving the computed rows' bits."""

import json
import sys

import numpy as np

import common as C
import hiprt as H
import refs as R
from hiprt import Buf, P

LAUNCHES = C.launches()  # the recording, then the derived launches (variants.py: the odd Markov steps' alignments)
rng = np.random.default_rng(1001)
C.log_open("t_markov")
V, RANK, WORLD = 129280, 256, 2
NC = V // WORLD
BN, SUB, BC, NP = 64, 4, 1024, 16
HEAD = (rng.standard_normal((V, RANK)) * 0.06).astype(np.float16)
EMB = C.rand_bf16(rng, (V, RANK), 1.0)
HEAD_B, EMB_B = Buf.of(HEAD), Buf.of(EMB)


def consts_of(case):
    return {k: (v if not isinstance(v, dict) else float(v["float"])) for k, v in case["consts"].items()}


def bias_ref(toks, seg0, ncols):
    """fp16(head[v] . fp16(emb[t])) for v in the segment: exact fp64 dot and its fp32 bound."""
    e = R.f16(C.from_bf16_bits(EMB[toks]))                             # [n, RANK]
    hseg = HEAD[seg0 * ncols:(seg0 + 1) * ncols].astype(np.float64)    # [ncols, RANK]
    return e @ hseg.T, C.gamma(RANK) * (np.abs(e) @ np.abs(hseg).T)


def run_bias(k, toks, t_stride, slot, seg0, out_rows=None, tok_off=0):
    """tok_off: TOK that many bytes into its buffer (8: an odd Markov step's out + 8 i)."""
    n = len(toks)
    tok_arr = np.zeros(max(n * t_stride, 1) + tok_off // 8, np.int64)
    tok_arr[tok_off // 8:][::t_stride][:n] = toks
    out = Buf((out_rows or n) * NC * 2)
    out.fill(0x7C)                                                     # 0x7C7C: fp16 +65408... a sentinel
    C.KIT.run("_bias", ((NC + BN * SUB - 1) // (BN * SUB), 1, 1), dict(
        TOK=P(Buf.of(tok_arr), "*i64", tok_off), t_stride=t_stride, SLOT=P(Buf.of(slot), "*i32"), EMB=P(EMB_B, "*bf16"),
        HEAD=P(HEAD_B, "*fp16"), OUT=P(out, "*fp16"), o_row=NC, n_rows=n, n_cols=NC, seg0=seg0), k)
    return out.download((out_rows or n, NC), np.float16)


def t_bias():
    none = np.full(V, -1, np.int32)
    for ci, c in enumerate(LAUNCHES["_bias"]):
        k = consts_of(c)
        n = c["scalars"]["n_rows"]
        ts = c["scalars"]["t_stride"]
        for seg0 in (0, 1):
            toks = rng.integers(0, V, n)
            slot = none.copy()
            hit = rng.random(n) < 0.3
            slot[toks[hit]] = 3                                        # rows whose token has a cached row: skipped
            got = run_bias(k, toks, ts, slot, seg0, tok_off=0 if c["tensors"]["TOK"][3] else 8)
            ref, bnd = bias_ref(toks, seg0, NC)
            miss = slot[toks] < 0
            if miss.any():
                m = C.cmp_rounded(got[miss].astype(np.float64), ref[miss], R.f16(ref[miss]), C.ulp_f16, bound=bnd[miss])
            else:
                m = dict(ok=True)
            m["hit_rows_untouched"] = bool(np.all(got[~miss].view(np.uint16) == 0x7C7C))
            m["ok"] = m["ok"] and m["hit_rows_untouched"]
            C.record(kernel="_bias", case=f"rec{ci} rows={n} stride={ts} seg0={seg0} misses={int(miss.sum())}", **m)


def t_score_finish_pick():
    kb = consts_of(LAUNCHES["_bias"][0])
    ks = consts_of(LAUNCHES["_score"][0])
    kf = consts_of(LAUNCHES["_finish"][0])
    kp = consts_of(LAUNCHES["_pick"][0])
    none = np.full(V, -1, np.int32)
    cached = rng.choice(V, 256, replace=False)
    for n, block, ts in ((1, 5, 6), (2, 5, 6), (3, 5, 4), (4, 5, 4)):
        steps = block
        lg_full = (rng.standard_normal((WORLD, n * block, NC)) * 2).astype(np.float32)   # [segs as ranks, rows, cols]
        toks = rng.integers(0, V, n)
        toks[0] = cached[5]                                                               # one cached token
        sends = []
        per_rank = {}
        for use_cache in (False, True):
            for seg0 in range(WORLD):
                # this rank's cache rows (Markov.fill: _bias of the cached tokens, NP rows a launch)
                if use_cache:
                    cache = np.zeros((256, NC), np.float16)
                    for r0 in range(0, 256, NP):
                        cache[r0:r0 + NP] = run_bias(kb, cached[r0:r0 + NP], 1, none, seg0)
                    slot = none.copy()
                    slot[cached] = np.arange(256, dtype=np.int32)
                else:
                    cache = np.zeros((1, NC), np.float16)
                    slot = none
                stage = run_bias(kb, toks, 1, slot, seg0)
                n_part = (NC + BC - 1) // BC
                pv, pi = Buf(n * n_part * 4), Buf(n * n_part * 4)
                tok_arr = np.zeros(n * ts, np.int64)
                tok_arr[::ts] = toks
                lg = lg_full[seg0]                                          # stream r's step i at row r * block + i
                i = 1
                base = lg.reshape(-1)[i * NC:]
                C.KIT.run("_score", (n_part, 1, 1), dict(
                    LG=P(Buf.of(np.ascontiguousarray(base)), "*fp32"), l_seg=n * block * NC, l_row=block * NC,
                    TOK=P(Buf.of(tok_arr), "*i64"), t_stride=ts, SLOT=P(Buf.of(slot), "*i32"),
                    CACHE=P(Buf.of(cache), "*fp16"), c_row=NC, STAGE=P(Buf.of(stage), "*fp16"), s_row=NC,
                    PV=P(pv, "*fp32"), PI=P(pi, "*i32"), n_rows=n, n_cols=NC, seg0=seg0, n_part=n_part), ks)
                # reference of the tile bests from the same bias bits (exact: one fp32 add, then a total order)
                bias_bits = np.where(slot[toks][:, None] >= 0, cache[np.maximum(slot[toks], 0)], stage).astype(np.float32)
                lrows = np.stack([lg[r * block + i] for r in range(n)])
                score = (lrows + bias_bits).astype(np.float32)
                gpv, gpi = pv.download((n, n_part), np.float32), pi.download((n, n_part), np.int32)
                wv = score.reshape(n, -1)
                okt = True
                for p in range(n_part):
                    seg = wv[:, p * BC:(p + 1) * BC]
                    j = np.argmax(seg, 1)
                    okt &= bool(np.array_equal(gpv[:, p], seg[np.arange(n), j]) and
                                np.array_equal(gpi[:, p], j + p * BC + seg0 * NC))
                C.record(kernel="_score", case=f"rows={n} seg0={seg0} cache={use_cache}", ok=okt, exact=True)
                send = Buf(n * 4 * 4)
                send.fill(0)
                out = Buf(n * ts * 8)
                C.KIT.run("_finish", (n, 1, 1), dict(PV=P(pv, "*fp32"), PI=P(pi, "*i32"), n_part=n_part,
                          OUT=P(out, "*i64"), o_stride=ts, SEND=P(send, "*fp32"), send=1), kf)
                sd = send.download((n, 4), np.float32)
                bi = np.argmax(wv, 1)
                okf = bool(np.array_equal(sd[:, 0], wv[np.arange(n), bi]) and
                           np.array_equal(sd[:, 1].astype(np.int64), bi + seg0 * NC) and np.all(sd[:, 2:] == 0))
                C.record(kernel="_finish", case=f"rows={n} seg0={seg0} cache={use_cache} (send)", ok=okf, exact=True)
                per_rank[(use_cache, seg0)] = (sd, score)
            # the ranks' bests gathered [WORLD, n, 4] -> the token
            g = np.stack([per_rank[(use_cache, s)][0] for s in range(WORLD)])
            out = Buf(n * ts * 8)
            out.fill(0)
            C.KIT.run("_pick", (n, 1, 1), dict(G=P(Buf.of(g), "*fp32"), n_rows=n, OUT=P(out, "*i64"), o_stride=ts), kp)
            got_tok = out.download(n * ts, np.int64)[::ts][:n]
            full = np.concatenate([per_rank[(use_cache, s)][1] for s in range(WORLD)], 1)   # [n, V]
            want_tok = np.argmax(full, 1)
            C.record(kernel="_pick", case=f"rows={n} cache={use_cache} (the TP2 step's token == full-vocab argmax)",
                     ok=bool(np.array_equal(got_tok, want_tok)))
            sends.append(got_tok)
            # against the fp64 bias (the model's definition): a changed token only within the bias error
            refb = []
            for s_ in range(WORLD):
                rb, bb = bias_ref(toks, s_, NC)
                refb.append((rb, bb))
            ref_full = np.concatenate([lrows_seg(lg_full, s_, n, block) + refb[s_][0] for s_ in range(WORLD)], 1)
            bnd_full = np.concatenate([refb[s_][1] + C.ulp_f16(refb[s_][0]) for s_ in range(WORLD)], 1)
            ref_tok = np.argmax(ref_full, 1)
            srt = np.sort(ref_full, 1)
            margin = srt[:, -1] - srt[:, -2]
            amb = margin <= 2 * bnd_full.max(1)
            ok = bool(np.all((got_tok == ref_tok) | amb))
            C.record(kernel="markov step", case=f"rows={n} cache={use_cache} vs fp64 bias", ok=ok,
                     same=int((got_tok == ref_tok).sum()), ambiguous=int(amb.sum()), rows=n)
        C.record(kernel="markov step", case=f"rows={n}: cached rows give the computed rows' drafts",
                 ok=bool(np.array_equal(sends[0], sends[1])))


def lrows_seg(lg_full, s, n, block, i=1):
    return np.stack([lg_full[s][r * block + i] for r in range(n)]).astype(np.float64)


def t_pick_nan_ties():
    kp = consts_of(LAUNCHES["_pick"][0])
    g = np.zeros((WORLD, 4, 4), np.float32)
    g[0, :, 0] = [1.0, np.nan, 2.0, 5.0]
    g[0, :, 1] = [10, 11, 12, 13]
    g[1, :, 0] = [1.0, 3.0, np.nan, 5.0]
    g[1, :, 1] = [70000, 70001, 70002, 70003]
    out = Buf(4 * 8)
    C.KIT.run("_pick", (4, 1, 1), dict(G=P(Buf.of(g), "*fp32"), n_rows=4, OUT=P(out, "*i64"), o_stride=1), kp)
    got = out.download(4, np.int64)
    # argmax order: NaN above all, then larger, ties to the lower index
    C.record(kernel="_pick", case="ties / NaN order", ok=bool(np.array_equal(got, [10, 11, 70002, 13])), got=got.tolist())


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for t in (t_bias, t_score_finish_pick, t_pick_nan_ties):
        if only and t.__name__ not in only:
            continue
        t()
    H.sync()
    sys.exit(C.done())
