"""Router / indexer-weight row matmuls and the MoE routing: _rowmm2 (fp32 and the indexer's bf16 weights), _rowmm2r,
_rowmm_parts, _route, _route_parts -- every recorded variant against refs.py; row invariance (a row's bits alone vs
inside a window) for the window kernels. The routing is graded by numerics.json router.noaux_tc (route_check: the
band rule for the picks, the weights clause), with logits from N(0, 1.5) down to the deep-negative rows of the first
tokens (every logit below -16.6) and over [-100, 30]; the score itself (_sqrt_softplus) against its 4 u32 bound on a
dense grid of that range."""

import ctypes
import sys

import numpy as np
import triton
import triton.language as tl

import common as C
import hiprt as H
import refs as R
from hiprt import Buf, P
from tri_kernels import _sqrt_softplus

LAUNCHES = C.launches()
rng = np.random.default_rng(777)
C.log_open("t_route")
K = 5120


def consts_of(case):
    return {k: (v if not isinstance(v, dict) else float(v["float"])) for k, v in case["consts"].items()}


def weights(n):
    return (rng.standard_normal((n, K)) * 0.02).astype(np.float16)


def dots_ref(x, w):
    xs = C.from_bf16_bits(x)
    ws = w.astype(np.float64)
    return xs @ ws.T, np.abs(xs) @ np.abs(ws).T


def run_rowmm2(fn, k, x, w, rows, n, wts=False, scale=1.0):
    bn = k["BN"]
    out = Buf(rows * n * (2 if wts else 4))
    args = dict(X=P(Buf.of(x), "*bf16"), xs=K, W=P(Buf.of(w), "*fp16"), OUT=P(out, "*bf16" if wts else "*fp32"),
                scale=scale)
    if fn == "_rowmm2r":
        args["rows"] = rows
        rb = k["RB"]
        grid = ((rows + rb - 1) // rb, (n + bn - 1) // bn, 1)
    else:
        grid = (rows, (n + bn - 1) // bn, 1)
    C.KIT.run(fn, grid, args, k)
    if wts:
        return C.from_bf16_bits(out.download((rows, n), np.uint16))
    return out.download((rows, n), np.float32).astype(np.float64)


def t_rowmm2():
    for ci, c in enumerate(LAUNCHES["_rowmm2"]):
        k = consts_of(c)
        n, wts = k["N"], bool(k["WTS"])
        scale = c["scalars"]["scale"]["float"]
        for rows in (1, 2) if not wts else (1, 2, 7, 16):
            x = C.rand_bf16(rng, (rows, K))
            w = weights(n)
            got = run_rowmm2("_rowmm2", k, x, w, rows, n, wts, scale)
            ref, aref = dots_ref(x, w)
            if wts:
                # (rowmm(x, w).to(bf16) * scale).to(bf16): two bf16 roundings; the first may flip by the dot's error
                d = C.gamma(K) * aref
                lo, hi = R.bf16(ref - d), R.bf16(ref + d)
                cand = [R.bf16(R.f32(v * scale)) for v in (lo, R.bf16(ref), hi)]
                ok = np.zeros(got.shape, bool)
                for cnd in cand:
                    ok |= got == cnd
                exact = got == cand[1]
                C.record(kernel="_rowmm2", case=f"rec{ci} WTS rows={rows}", ok=bool(ok.all()),
                         frac_not_nearest=float(1 - exact.mean()), n=int(got.size))
            else:
                m = C.cmp_bound(got, ref, C.gamma(K) * aref)
                C.record(kernel="_rowmm2", case=f"rec{ci} fp32 rows={rows}", **m)


def t_rowmm2r():
    # rows 3..16 take _rowmm2r (RB = rows); each row's bits must equal _rowmm2's 1-row launch (row invariance)
    c2 = next(c for c in LAUNCHES["_rowmm2"] if not c["consts"]["WTS"])
    for ci, c in enumerate(LAUNCHES["_rowmm2r"]):
        k = consts_of(c)
        n, rows = k["N"], k["RB"]
        x = C.rand_bf16(rng, (rows, K))
        w = weights(n)
        got = run_rowmm2("_rowmm2r", k, x, w, rows, n)
        ref, aref = dots_ref(x, w)
        m = C.cmp_bound(got, ref, C.gamma(K) * aref)
        C.record(kernel="_rowmm2r", case=f"rec{ci} rows={rows}", **m)
        solo = np.concatenate([run_rowmm2("_rowmm2", consts_of(c2), x[r:r + 1], w, 1, n) for r in range(rows)])
        C.record(kernel="_rowmm2r", case=f"rec{ci} rows={rows} == _rowmm2 per row (bits)",
                 ok=bool(np.array_equal(solo.view(np.uint64), got.view(np.uint64))),
                 differing=int((solo != got).sum()))


def run_parts(k, x, w, rows, n):
    bn, cpg = k["BN"], k["CPG"]
    kc = K // 256
    s = Buf(rows * kc * n * 4)
    args = dict(X=P(Buf.of(x), "*bf16"), xs=K, W=P(Buf.of(w), "*fp16"), S=P(s, "*fp32"), rows=rows)
    C.KIT.run("_rowmm_parts", ((n + bn - 1) // bn, kc // cpg, 1), args, k)
    return s.download((rows, kc, n), np.float32).astype(np.float64)


def t_rowmm_parts():
    by_n = {}
    for ci, c in enumerate(LAUNCHES["_rowmm_parts"]):
        k = consts_of(c)
        n, rows = k["N"], k["RB"]
        x = C.rand_bf16(rng, (rows, K))
        w = by_n.setdefault(n, weights(n))
        got = run_parts(k, x, w, rows, n)
        xs = C.from_bf16_bits(x).reshape(rows, K // 256, 256)
        ws = w.astype(np.float64).reshape(n, K // 256, 256)
        ref = np.einsum("rtk,ntk->rtn", xs, ws)
        aref = np.einsum("rtk,ntk->rtn", np.abs(xs), np.abs(ws))
        m = C.cmp_bound(got, ref, C.gamma(256) * aref)
        C.record(kernel="_rowmm_parts", case=f"rec{ci} N={n} rows={rows} BN={k['BN']}", **m)
        if rows > 1:
            one = next((cc for cc in LAUNCHES["_rowmm_parts"] if cc["consts"]["N"] == n and consts_of(cc)["RB"] == 1),
                       None)
            k1 = consts_of(one) if one else None
            if k1 is None:
                kv = C.extra_variant("_rowmm_parts", c, RB=1)
            same = True
            for r in range(rows):
                if k1 is not None:
                    s1 = run_parts(k1, x[r:r + 1], w, 1, n)
                else:
                    sb = Buf(K // 256 * n * 4)
                    C.run_variant(kv, ((n + k["BN"] - 1) // k["BN"], K // 256, 1), dict(
                        X=P(Buf.of(x[r:r + 1]), "*bf16"), xs=K, W=P(Buf.of(w), "*fp16"), S=P(sb, "*fp32"), rows=1))
                    s1 = sb.download((1, K // 256, n), np.float32).astype(np.float64)
                same &= bool(np.array_equal(s1[0], got[r]))
            C.record(kernel="_rowmm_parts", case=f"row invariance N={n} rows={rows} (vs 1-row launch"
                     f"{'' if k1 else ', RB=1 extra variant: N=128 has no recorded 1-row tile'})", ok=same)


U32 = 2.0 ** -24


def score_bound(lg, dlg):
    """numerics.json router.noaux_tc (1.6.0; dsref/bounds.py softplus_sqrt_bound): the exact score sqrt(softplus(l)) of
    logits l (fp64) and the bound on an fp32 score's error, |dl| sigmoid(l) / (2 score) + 4 u32 score, where dl bounds
    the logits' own error (0 when the kernel received them exactly)."""
    lg = np.asarray(lg, np.float64)
    sc = np.sqrt(np.log1p(np.exp(-np.abs(lg))) + np.maximum(lg, 0))
    sig = np.exp(-np.logaddexp(0.0, -lg))
    return sc, dlg * sig / (2 * np.maximum(sc, 1e-300)) + 4 * U32 * sc


def selection(sel, band, ref_picks, topk):
    """numerics.json router.noaux_tc procedure step 4, per row: c = the k-th largest reference biased score, b_c its
    band; MUST = {e: sel_e > c + b_e + b_c}, FREE = {e: |sel_e - c| <= b_e + b_c} minus MUST. Returns (must, free)."""
    kth = ref_picks[:, topk - 1]
    c = np.take_along_axis(sel, kth[:, None], 1)
    bc = np.take_along_axis(band, kth[:, None], 1)
    must = sel > c + band + bc
    free = (np.abs(sel - c) <= band + bc) & ~must
    return must, free


def set_ok(must, free, got, topk):
    """A row's picks under the band rule: topk distinct experts, every MUST one, none outside MUST and FREE."""
    ne = must.size
    if len(set(got.tolist())) != topk or got.min() < 0 or got.max() >= ne:
        return False
    g = np.zeros(ne, bool)
    g[got] = True
    return bool(np.all(g[must]) and np.all((must | free)[g]))


def route_check(name, case, picks, wts, logits, bias, topk, slots, shared_id, scale, dlg):
    """picks [R, slots] (topk routed by score + bias, then shared_id) and wts [R, slots] against numerics.json
    router.noaux_tc, on the logits the kernel saw (fp64 [R, NE] of fp32 values; dlg their own error bound, 0 when the
    kernel received them exactly). Selection: the band rule (selection()) with b_e = 2 (score bound + u32 |score +
    bias|). Weights: within scale (sum |d score| / S + |dS| score / S^2) + 4 u32 w of the reference weights of the GPU's
    own picks (S their reference score sum, the 1e-20 of the kernel's normalisation kept in the reference)."""
    rows, ne = logits.shape
    ref_picks, _, _, margin = R.gate(logits, bias, topk, scale)
    sc, dsc = score_bound(logits, dlg)
    sel = sc + bias.astype(np.float64)
    band = 2 * (dsc + U32 * np.abs(sel))
    must, free = selection(sel, band, ref_picks, topk)
    same_set = reordered = ambiguous = bad = 0
    for r in range(rows):
        got = picks[r, :topk]
        if np.array_equal(np.sort(got), np.sort(ref_picks[r])):
            same_set += 1
            reordered += int(not np.array_equal(got, ref_picks[r]))
        elif set_ok(must[r], free[r], got, topk):
            ambiguous += 1
        else:
            bad += 1
    shared_ok = bool(np.all(picks[:, topk:] == shared_id)) and bool(np.all(wts[:, topk:] == 1.0))
    gp = np.clip(picks[:, :topk], 0, ne - 1)
    scp = np.take_along_axis(sc, gp, 1)
    s = scp.sum(1, keepdims=True)
    w_ref = scp / (s + 1e-20) * scale
    dsp = np.take_along_axis(dsc, gp, 1)
    w_bound = scale * (dsp.sum(1, keepdims=True) / s + dsp.sum(1, keepdims=True) * scp / s ** 2) + 4 * U32 * w_ref
    mw = C.cmp_bound(wts[:, :topk], w_ref, w_bound)
    C.record(kernel=name, case=case, ok=bool(bad == 0 and shared_ok and mw["ok"]), rows=rows, same_set_rows=same_set,
             reordered_rows=reordered, ambiguous_rows=ambiguous, unexplained_rows=bad, shared_slot_ok=shared_ok,
             w_max_err_over_bound=mw["max_err_over_bound"], min_margin=float(margin.min()),
             min_logit=float(logits.min()), max_logit=float(logits.max()))


# logit regimes beyond N(0, 1.5): the first tokens' rows at deep layers (every logit below -16.6, where the upstream
# formula's 1 + exp(lg) rounded to 1 and every expert scored only its bias), the whole [-100, 30] range, and the
# moderately negative logits where log(1 + e) lost precision
REGIMES = {"normal(0,1.5)": lambda n: rng.standard_normal(n) * 1.5,
           "all below -16.6 (U(-40,-17))": lambda n: rng.uniform(-40, -17, n),
           "wide U(-100,30)": lambda n: rng.uniform(-100, 30, n),
           "negative normal(-12,4)": lambda n: rng.standard_normal(n) * 4 - 12}


def t_route():
    for ci, c in enumerate(LAUNCHES["_route"]):
        k = consts_of(c)
        ne, topk, slots = k["NE"], k["TOPK"], k["SLOTS"]
        scale, shared_id = c["scalars"]["scale"]["float"], c["scalars"]["shared_id"]
        for regime, draw in REGIMES.items():
            for rows in (1, 17, 256):
                lg = draw((rows, ne)).astype(np.float32)
                bias = (rng.standard_normal(ne) * 0.1).astype(np.float32)
                pick, wts = Buf(rows * slots * 4), Buf(rows * slots * 4)
                C.KIT.run("_route", (rows, 1, 1), dict(L=P(Buf.of(lg), "*fp32"), BIAS=P(Buf.of(bias), "*fp32"),
                          PICK=P(pick, "*i32"), WTS=P(wts, "*fp32"), scale=scale, shared_id=shared_id), k)
                route_check("_route", f"rec{ci} NE={ne} rows={rows} logits {regime}",
                            pick.download((rows, slots), np.int32),
                            wts.download((rows, slots), np.float32).astype(np.float64), lg.astype(np.float64), bias,
                            topk, slots, shared_id, scale, 0.0)


def t_route_parts():
    for ci, c in enumerate(LAUNCHES["_route_parts"]):
        k = consts_of(c)
        ne, topk, slots, kc = k["NE"], k["TOPK"], k["SLOTS"], k["KC"]
        scale, shared_id = c["scalars"]["scale"]["float"], c["scalars"]["shared_id"]
        rows = c["tensors"]["S"][1][0]
        for regime, draw in REGIMES.items():
            # the chunk sums of a row whose logits fall in the regime
            s = (draw((rows, 1, ne)) / kc + rng.standard_normal((rows, kc, ne)) * 0.35).astype(np.float32)
            bias = (rng.standard_normal(ne) * 0.1).astype(np.float32)
            pick, wts = Buf(rows * slots * 4), Buf(rows * slots * 4)
            C.KIT.run("_route_parts", (rows, 1, 1), dict(S=P(Buf.of(s), "*fp32"), BIAS=P(Buf.of(bias), "*fp32"),
                      PICK=P(pick, "*i32"), WTS=P(wts, "*fp32"), scale=scale, shared_id=shared_id), k)
            lg = s.astype(np.float64).sum(1)
            dlg = C.gamma(kc) * np.abs(s.astype(np.float64)).sum(1)
            route_check("_route_parts", f"rec{ci} NE={ne} rows={rows} logits {regime}",
                        pick.download((rows, slots), np.int32),
                        wts.download((rows, slots), np.float32).astype(np.float64), lg, bias, topk, slots, shared_id,
                        scale, dlg)


@triton.jit
def _t_score(L, OUT, n, BLOCK: tl.constexpr):
    """Test-only: tri_kernels._sqrt_softplus, the score _route and _route_parts inline, of each element."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(OUT + i, _sqrt_softplus(tl.load(L + i, mask=m, other=0.0)), mask=m)


def t_score_grid():
    """The score alone against numerics.json's score bound on exact fp32 logits (4 u32 score) over lg in [-100, 30]:
    a 2^-14 grid, 4M uniform draws, and every fp32 value within 4096 ulps of the formula's switch points (-16, 20)
    and of -16.6 and -87.3 (where upstream's 1 + e and exp(lg) underflowed)."""
    f, block, shared, _mod = C.jit_kernel(_t_score, {"L": "*fp32", "OUT": "*fp32", "n": "i32", "BLOCK": "constexpr"},
                                         {"BLOCK": 256}, num_warps=8)
    near = [np.float32(v).view(np.int32) + np.arange(-4096, 4097, dtype=np.int32) for v in (-87.3, -16.6, -16.0, 20.0)]
    lg = np.concatenate([np.arange(-100.0, 30.0, 2.0 ** -14), rng.uniform(-100, 30, 1 << 22)]
                        + [n.view(np.float32) for n in near]).astype(np.float32)
    n = lg.size
    out = Buf(n * 4)
    lb = Buf.of(lg)
    H.launch(f, ((n + 255) // 256, 1, 1), (block, 1, 1), shared,
             [ctypes.c_uint64(lb.ptr), ctypes.c_uint64(out.ptr), ctypes.c_int32(n), ctypes.c_uint64(0),
              ctypes.c_uint64(0)])
    got = out.download((n,), np.float32).astype(np.float64)
    l64 = lg.astype(np.float64)
    sc, bound = score_bound(l64, 0.0)
    for lo, hi in ((-100, -87.3), (-87.3, -16.6), (-16.6, -16), (-16, -2), (-2, 0), (0, 20), (20, 30)):
        sl = (l64 >= lo) & (l64 < hi)
        m = C.cmp_bound(got[sl], sc[sl], bound[sl])
        C.record(kernel="_sqrt_softplus (_route, _route_parts)", case=f"score grid lg in [{lo}, {hi})", **m,
                 max_err_u32_rel=float((np.abs(got[sl] - sc[sl]) / sc[sl]).max() / U32),
                 zero_scores=int((got[sl] == 0).sum()))


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for t in (t_rowmm2, t_rowmm2r, t_rowmm_parts, t_route, t_route_parts, t_score_grid):
        if only and t.__name__ not in only:
            continue
        t()
    H.sync()
    sys.exit(C.done())
