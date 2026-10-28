"""k-dec2: hc_pre2's decode-window chain after _hc_post_only (hc_dots path: _hc_finish_s's collapse + RMSNorm,
_hc_mix_part of the stored streams, _hc_sinkhorn) as one launch, from tri_kernels.py's own code so each part keeps its
arithmetic: programs (g, b < NB, mg) are _hc_mix_part's (its rows' partial dots, _hc_mix_rows unchanged); the last of a
row group's 3 * NB mix programs to arrive (an acq_rel counter, reset after) runs _hc_sinkhorn's code for the group's
rows; programs (g, NB, 0) run _hc_finish_s's role 1 (the collapse with the carried pre-mix, then the RMSNorm) for them,
beside the mixes. Compiled with 8 warps, as _hc_finish_s and _hc_sinkhorn are (_hc_mix_part: 2; the mixes' per-mix
reduction is one warp's 32 lanes x 4 columns in either layout, which the bit test checks)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "tfp/tri/src"))
import triton
import triton.language as tl
import tri_kernels as K


@triton.jit
def _sink_row(PART, BASE, SCALE, PRE_OUT, POST, COMB, eps, hc_eps, r, D: tl.constexpr, NB: tl.constexpr,
              ITERS: tl.constexpr):
    """_hc_sinkhorn's code for row r."""
    m = tl.arange(0, 32)
    s0 = tl.load(SCALE + 0)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)
    base = tl.load(BASE + m, mask=m < 24, other=0.0)
    mix = tl.zeros((32,), dtype=tl.float32)
    ss = 0.0
    for b in tl.static_range(NB):
        mix += tl.load(PART + (r * NB + b) * 32 + m)
        ss += tl.load(PART + (r * NB + b) * 32 + 24)
    mix = mix * (1.0 / tl.sqrt(ss / (4 * D) + eps))
    sv = tl.arange(0, 4)
    pre_l = tl.sum(tl.where(m[None, :] == sv[:, None], (mix * s0 + base)[None, :], 0.0), axis=1)
    post_l = tl.sum(tl.where(m[None, :] == (sv[:, None] + 4), (mix * s1 + base)[None, :], 0.0), axis=1)
    pre = 1.0 / (1.0 + tl.exp(-pre_l)) + hc_eps
    post = 2.0 * (1.0 / (1.0 + tl.exp(-post_l)))
    ii = tl.arange(0, 4)[:, None]
    jj = tl.arange(0, 4)[None, :]
    flat = 8 + ii * 4 + jj
    cl = tl.sum(tl.where(m[None, None, :] == flat[:, :, None], (mix * s2 + base)[None, None, :], 0.0), axis=2)
    cmax = tl.max(cl, axis=1)
    ce = tl.exp(cl - cmax[:, None])
    comb = ce / tl.sum(ce, axis=1)[:, None] + hc_eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    for _ in range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    tl.store(PRE_OUT + r * 4 + sv, pre)
    tl.store(POST + r * 4 + sv, post)
    tl.store(COMB + r * 16 + ii * 4 + jj, comb)


@triton.jit
def _fin_row(X, PRE_IN, NW, OUT, eps, r, D: tl.constexpr, BLOCK: tl.constexpr):
    """_hc_finish_s's role 1 (NROT 0) for row r: the collapse with the carried pre-mix, the RMSNorm, the stored row."""
    pl = tl.arange(0, BLOCK // 8)
    K._prefetch_l2(NW + tl.minimum(pl * 64, D - 1))
    p0 = tl.load(PRE_IN + r * 4 + 0)
    p1 = tl.load(PRE_IN + r * 4 + 1)
    p2 = tl.load(PRE_IN + r * 4 + 2)
    p3 = tl.load(PRE_IN + r * 4 + 3)
    xs = ()
    for c in tl.static_range(D // BLOCK):
        d = c * BLOCK + tl.arange(0, BLOCK)
        xs = xs + (tl.load(X + r * (4 * D) + d), tl.load(X + r * (4 * D) + D + d),
                   tl.load(X + r * (4 * D) + 2 * D + d), tl.load(X + r * (4 * D) + 3 * D + d))
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    cs = ()
    for c in tl.static_range(D // BLOCK):
        x0 = xs[4 * c].to(tl.float32)
        x1 = xs[4 * c + 1].to(tl.float32)
        x2 = xs[4 * c + 2].to(tl.float32)
        x3 = xs[4 * c + 3].to(tl.float32)
        cc = (((p0 * x0 + p1 * x1) + p2 * x2) + p3 * x3).to(tl.bfloat16).to(tl.float32)
        acc += cc * cc
        cs = cs + (cc,)
    rinv = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / D + eps)
    K._rot_out(OUT, r, cs, rinv, NW, OUT, OUT, 0, D, BLOCK, False)


@triton.jit
def _ss_tree(ss):
    """tl.sum(ss) as _hc_mix_part's 2-warp layout reduces it (ss [128]: 2 elements a thread, 32 lanes, 2 warps): each
    thread's pair, then a 16-lane row's DPP row_shr 8 / 4 / 2 / 1 tree (lane 15), the two rows (permlanex16), the two
    warps. Every step adds two terms (no order inside a pair), so the tree alone sets the bits."""
    a = tl.sum(tl.reshape(ss, (64, 2)), axis=1)               # thread t: elements 2t, 2t + 1
    a = tl.reshape(a, (4, 2, 8))                              # (warp row, lane j >= 8 / j < 8, j % 8)
    a = tl.sum(a, axis=1)                                     # row_shr 8: lanes 8..15 hold a[j] + a[j - 8]
    a = tl.sum(tl.reshape(a, (4, 2, 4)), axis=1)              # row_shr 4
    a = tl.sum(tl.reshape(a, (4, 2, 2)), axis=1)              # row_shr 2
    a = tl.sum(tl.reshape(a, (4, 2)), axis=1)                 # row_shr 1: the row's sum in its lane 15
    a = tl.sum(tl.reshape(a, (2, 2)), axis=1)                 # permlanex16: the warp's two rows
    return tl.sum(a, axis=0)                                  # the two warps


@triton.jit
def _mix_rows8(X, PART, ws, r, b, mg, m, k, WIDE: tl.constexpr, D: tl.constexpr, NB: tl.constexpr, KB: tl.constexpr,
               SUB: tl.constexpr, MB: tl.constexpr):
    """_hc_mix_rows (POSTED False) with its sum of squares reduced in _hc_mix_part's tree (_ss_tree)."""
    s = (b * KB) // D
    col0 = b * KB - s * D
    acc = tl.zeros((MB,), dtype=tl.float32)
    ss = tl.zeros((SUB,), dtype=tl.float32)
    for t in tl.static_range(KB // SUB):
        d = col0 + t * SUB + k
        x = tl.load(X + r * WIDE + s * D + d).to(tl.float32)
        acc += tl.sum(ws[t] * x[None, :], axis=1)
        ss = tl.fma(x, x, ss)
    tl.store(PART + (r * NB + b) * 32 + m, acc)
    if mg == 0:
        tl.store(PART + (r * NB + b) * 32 + 24, _ss_tree(ss))


@triton.jit
def _hc_fused(X, FN, PART, CNT, BASE, SCALE, PRE_IN, NW, OUT, PRE_OUT, POST, COMB, eps, hc_eps, rows,
              WIDE: tl.constexpr, D: tl.constexpr, NB: tl.constexpr, SUB: tl.constexpr, MB: tl.constexpr,
              RB: tl.constexpr, ITERS: tl.constexpr, BLOCK: tl.constexpr, SST: tl.constexpr = False):
    g = tl.program_id(0)
    b = tl.program_id(1)
    mg = tl.program_id(2)
    r0 = g * RB
    if b < NB:
        KB: tl.constexpr = WIDE // NB
        m = mg * MB + tl.arange(0, MB)
        k = tl.arange(0, SUB)
        ws = ()
        for t in tl.static_range(KB // SUB):
            ws = ws + (tl.load(FN + m[:, None] * WIDE + b * KB + t * SUB + k[None, :]),)
        for i in tl.static_range(RB):
            if r0 + i < rows:
                if SST:
                    _mix_rows8(X, PART, ws, r0 + i, b, mg, m, k, WIDE, D, NB, KB, SUB, MB)
                else:
                    K._hc_mix_rows(X, X, X, 0, POST, COMB, PART, ws, r0 + i, b, mg, m, k, WIDE, D, NB, KB, SUB, MB, 1, False)
        tl.debug_barrier()
        old = tl.atomic_add(CNT + g, 1, sem="acq_rel")
        if old == 3 * NB - 1:
            for i in tl.static_range(RB):
                if r0 + i < rows:
                    _sink_row(PART, BASE, SCALE, PRE_OUT, POST, COMB, eps, hc_eps, r0 + i, D, NB, ITERS)
            tl.atomic_xchg(CNT + g, 0, sem="relaxed")
    else:
        if mg == 0:
            for i in tl.static_range(RB):
                if r0 + i < rows:
                    _fin_row(X, PRE_IN, NW, OUT, eps, r0 + i, D, BLOCK)


def hc_rb(rows):
    t = [1, 2, 3, 2, 3, 2, 4, 4]
    return t[rows - 1] if 1 <= rows <= 8 else (4 if rows <= 12 else 8)
