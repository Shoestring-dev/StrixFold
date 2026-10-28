"""Row kernels around the layers: _rmsnorm, _rmsnorm_rot, _kv_norm_rope, _q_kv_norm, _rope_heads, _embed_init, _tap,
_collapse, _collapse_norm, _engram_gate, _hc_post -- every recorded variant, on the AOT kit, against refs.py."""

import copy
import json
import sys

import numpy as np

import common as C
import hiprt as H
import refs as R
from hiprt import Buf, P

LAUNCHES = C.launches()
NREC = {fn: len(cs) for fn, cs in json.loads((C.ROOT / "upstream" / "launches.json").read_text()).items()}
EPS = 1e-20
rng = np.random.default_rng(1234)
C.log_open("t_basic")


def consts_of(case):
    return {k: (v if not isinstance(v, dict) else float(v["float"])) for k, v in case["consts"].items()}


def rope_tables(npos, rd, theta=10000.0):
    inv = 1.0 / theta ** (np.arange(0, rd, 2) / rd)
    ang = np.outer(np.arange(npos), inv)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


COS, SIN = rope_tables(70000, 64)
cos_b, sin_b = Buf.of(COS), Buf.of(SIN)


def i32(x):
    return int(x)


# ---- _rmsnorm ---------------------------------------------------------------------------------------------------
def t_rmsnorm():
    for ci, c in enumerate(LAUNCHES["_rmsnorm"]):
        k = consts_of(c)
        D = k["D"]
        for rows in sorted({c["tensors"]["X"][1][0], 1, 3}):
            x = C.rand_bf16(rng, (rows, D), 2.0)
            w = C.rand_bf16(rng, (D,), 0.5)
            out = Buf(rows * D * 2)
            xb, wb = Buf.of(x), Buf.of(w)
            C.KIT.run("_rmsnorm", (rows, 1, 1), dict(X=P(xb, "*bf16"), xs=D, W=P(wb, "*bf16"), OUT=P(out, "*bf16"),
                                                     os_=D, eps=EPS), k)
            got = C.from_bf16_bits(out.download((rows, D), np.uint16))
            xs, ws = C.from_bf16_bits(x), C.from_bf16_bits(w)
            exact = ws * xs / np.sqrt(np.mean(xs * xs, -1, keepdims=True) + EPS)
            m = C.cmp_rounded(got, exact, R.bf16(exact), C.ulp_bf16)
            C.record(kernel="_rmsnorm", case=f"rec{ci} D={D} rows={rows}", **m)


# ---- _rmsnorm_rot / _q_kv_norm (rotations) ------------------------------------------------------------------------
def t_rmsnorm_rot():
    for ci, c in enumerate(LAUNCHES["_rmsnorm_rot"]):
        k = consts_of(c)
        D, nrot = k["D"], k["NROT"]
        for rows in (1, c["tensors"]["X"][1][0]):
            x = C.rand_bf16(rng, (rows, D), 1.0)
            w = C.rand_bf16(rng, (D,), 0.3)
            suh = [np.sign(rng.standard_normal(D)).astype(np.float16) * np.float16(1.0) for _ in range(2)]
            xb, wb, out = Buf.of(x), Buf.of(w), Buf(rows * D * 2)
            sb = [Buf.of(s) for s in suh]
            hb = [Buf(rows * D * 2) for _ in range(2)]
            args = dict(X=P(xb, "*bf16"), xs=D, W=P(wb, "*bf16"), OUT=P(out, "*bf16"), os_=D, eps=EPS)
            for i in range(2):
                if i < nrot:
                    args[f"S{i}"], args[f"H{i}"] = P(sb[i], "*fp16"), P(hb[i], "*fp16")
                else:
                    args[f"S{i}"], args[f"H{i}"] = P(out, "*bf16"), P(out, "*bf16")
            C.KIT.run("_rmsnorm_rot", (rows, 1, 1), args, k)
            got = C.from_bf16_bits(out.download((rows, D), np.uint16))
            xs, ws = C.from_bf16_bits(x), C.from_bf16_bits(w)
            exact = ws * xs / np.sqrt(np.mean(xs * xs, -1, keepdims=True) + EPS)
            m = C.cmp_rounded(got, exact, R.bf16(exact), C.ulp_bf16)
            C.record(kernel="_rmsnorm_rot", case=f"rec{ci} norm rows={rows}", **m)
            # the rotation of the kernel's own bf16 row (the definition: rot of the stored normed row)
            for i in range(nrot):
                gh = hb[i].download((rows, D), np.float16).astype(np.float64)
                ex = (got * suh[i].astype(np.float64)).reshape(rows, D // 128, 128) @ R.H128.T / np.sqrt(128)
                ex = ex.reshape(rows, D)
                m = C.cmp_rounded(gh, ex, R.f16(ex), C.ulp_f16)
                C.record(kernel="_rmsnorm_rot", case=f"rec{ci} rot{i} rows={rows}", **m)


def kv_inputs(rows, DK, ring_rows):
    y = C.rand_bf16(rng, (rows, DK), 3.0)
    wk = C.rand_bf16(rng, (DK,), 0.4)
    pos = rng.integers(0, 60000, rows).astype(np.int64)
    slots = rng.permutation(ring_rows)[:rows].astype(np.int64)
    slots[rng.random(rows) < 0.25] = -1
    return y, wk, pos, slots


def kv_ref_prequant(y, wk, pos, rd):
    """The window KV row before its FP8 step: RMSNorm (bf16), RoPE of the last rd dims (exact, before bf16)."""
    ys, ws = C.from_bf16_bits(y), C.from_bf16_bits(wk)
    n = R.rmsnorm(ys, ws, EPS)
    ex = n.copy()
    ex[..., -rd:] = R.rope_pairs(n[..., -rd:], COS[pos].astype(np.float64), SIN[pos].astype(np.float64))
    return ex


def check_kv(name, case, got_q, got_plain, ring_got, ring_init, y, wk, pos, rd, slots, quant):
    """got_plain: the same kernel with QUANT off (its bf16 pre-quant row) against the fp64 reference; got_q: the
    quantized row, which must be exactly the reference FP8 step applied to got_plain; the ring rows exact."""
    ex = kv_ref_prequant(y, wk, pos, rd)
    m = C.cmp_rounded(got_plain, ex, R.bf16(ex), C.ulp_bf16, max_ulp=2)
    C.record(kernel=name, case=case + " prequant(QUANT=0 variant)", **m)
    want_q = R.bf16(R.fp8_qd_blocks(got_plain, 32)) if quant else got_plain
    exact_q = bool(np.array_equal(got_q, want_q))
    want = ring_init.copy()
    for r, s_ in enumerate(slots):
        if s_ >= 0:
            want[s_] = got_q[r]
    ring_exact = bool(np.array_equal(ring_got, want))
    C.record(kernel=name, case=case + " fp8 step + ring", ok=exact_q and ring_exact, quant_exact=exact_q,
             ring_exact=ring_exact, mismatches=int((got_q != want_q).sum()))


def t_kv_norm_rope():
    for ci, c in enumerate(LAUNCHES["_kv_norm_rope"]):
        k = consts_of(c)
        DK, rd, quant = k["D"], k["RD"], bool(k["QUANT"])
        ring_rows = c["scalars"]["ring_size"]
        derived = ci >= NREC["_kv_norm_rope"]
        for rows in sorted({1, c["tensors"]["Y"][1][0]} | ({17, ring_rows} if derived else set())):
            rows = min(rows, ring_rows)
            y, wk, pos, slots = kv_inputs(rows, DK, ring_rows)
            ring_init = C.rand_bf16(rng, (ring_rows, DK))
            yb, wb, sb = Buf.of(y), Buf.of(wk), Buf.of(slots)
            pb, po = pos_buf(c, pos)
            tag = f"{'derived' if derived else 'rec'}{ci} rows={rows} POS aligned={po == 0}"
            outs = {}
            for q in (True, False):
                ringb, out = Buf.of(ring_init), Buf(rows * DK * 2)
                args = dict(Y=P(yb, "*bf16"), W=P(wb, "*bf16"), COS=P(cos_b, "*fp32"), SIN=P(sin_b, "*fp32"),
                            POS=P(pb, "*i64", po), OUT=P(out, "*bf16"), RING=P(ringb, "*bf16"), SLOT_OF=P(sb, "*i64"),
                            ring_size=ring_rows, eps=EPS)
                if q == quant and derived:
                    if not C.run_exact("_kv_norm_rope", (rows, 1, 1), args, k, tag):
                        break
                elif q == quant:
                    C.KIT.run("_kv_norm_rope", (rows, 1, 1), args, k)
                else:
                    kv = copy.deepcopy(c)
                    kv["tensors"]["POS"][3] = po == 0
                    C.run_variant(C.extra_variant("_kv_norm_rope", kv, QUANT=q), (rows, 1, 1), args)
                outs[q] = (C.from_bf16_bits(out.download((rows, DK), np.uint16)),
                           C.from_bf16_bits(ringb.download((ring_rows, DK), np.uint16)))
            if len(outs) < 2:
                continue
            check_kv("_kv_norm_rope", tag, outs[True][0], outs[False][0], outs[True][1],
                     C.from_bf16_bits(ring_init), y, wk, pos, rd, slots, True)


def t_q_kv_norm():
    for ci, c in enumerate(LAUNCHES["_q_kv_norm"]):
        k = consts_of(c)
        D, DK, rd, nrot, split = k["D"], k["DK"], k["RD"], k["NROT"], bool(k["SPLIT"])
        ring_rows = c["scalars"]["ring_size"]
        for rows in (1, 6, 16):
            x = C.rand_bf16(rng, (rows, D), 1.0)
            w = C.rand_bf16(rng, (D,), 0.3)
            suh = [np.sign(rng.standard_normal(D)).astype(np.float16) for _ in range(2)]
            y, wk, pos, slots = kv_inputs(rows, DK, ring_rows)
            ring_init = C.rand_bf16(rng, (ring_rows, DK))
            xb, wb, out = Buf.of(x), Buf.of(w), Buf(rows * D * 2)
            sb = [Buf.of(s) for s in suh]
            hb = [Buf(rows * D * 2) for _ in range(2)]
            yb, wkb, pb, slb = Buf.of(y), Buf.of(wk), Buf.of(pos), Buf.of(slots)
            ringb, kout = Buf.of(ring_init), Buf(rows * DK * 2)
            args = dict(X=P(xb, "*bf16"), xs=D, W=P(wb, "*bf16"), OUT=P(out, "*bf16"), os_=D, eps=EPS,
                        Y=P(yb, "*bf16"), WK=P(wkb, "*bf16"), COS=P(cos_b, "*fp32"), SIN=P(sin_b, "*fp32"),
                        POS=P(pb, "*i64"), KOUT=P(kout, "*bf16"), RING=P(ringb, "*bf16"), SLOT_OF=P(slb, "*i64"),
                        ring_size=ring_rows)
            for i in range(2):
                if i < nrot:
                    args[f"S{i}"], args[f"H{i}"] = P(sb[i], "*fp16"), P(hb[i], "*fp16")
                else:
                    args[f"S{i}"], args[f"H{i}"] = P(out, "*bf16"), P(out, "*bf16")
            progs = 2 + (nrot if split else 0)
            C.KIT.run("_q_kv_norm", (rows, progs, 1), args, k)
            got = C.from_bf16_bits(out.download((rows, D), np.uint16))
            xs, ws = C.from_bf16_bits(x), C.from_bf16_bits(w)
            exact = ws * xs / np.sqrt(np.mean(xs * xs, -1, keepdims=True) + EPS)
            m = C.cmp_rounded(got, exact, R.bf16(exact), C.ulp_bf16)
            C.record(kernel="_q_kv_norm", case=f"rec{ci} q-norm rows={rows}", **m)
            for i in range(nrot):
                gh = hb[i].download((rows, D), np.float16).astype(np.float64)
                ex = ((got * suh[i].astype(np.float64)).reshape(rows, D // 128, 128) @ R.H128.T / np.sqrt(128)).reshape(rows, D)
                m = C.cmp_rounded(gh, ex, R.f16(ex), C.ulp_f16)
                C.record(kernel="_q_kv_norm", case=f"rec{ci} rot{i} rows={rows}", **m)
            gk = C.from_bf16_bits(kout.download((rows, DK), np.uint16))
            ring_got = C.from_bf16_bits(ringb.download((ring_rows, DK), np.uint16))
            # the pre-quant row from the QUANT=0 variant of this kernel on the same inputs
            ring2, kout2 = Buf.of(ring_init), Buf(rows * DK * 2)
            args2 = dict(args, KOUT=P(kout2, "*bf16"), RING=P(ring2, "*bf16"))
            C.run_variant(C.extra_variant("_q_kv_norm", c, QUANT=False), (rows, progs, 1), args2)
            gplain = C.from_bf16_bits(kout2.download((rows, DK), np.uint16))
            check_kv("_q_kv_norm", f"rec{ci} kv rows={rows}", gk, gplain, ring_got, C.from_bf16_bits(ring_init),
                     y, wk, pos, rd, slots, bool(k["QUANT"]))


# ---- _rope_heads ------------------------------------------------------------------------------------------------
def pos_buf(case, pos):
    """The positions as the case passes them: at the buffer's start (16-byte aligned), or 8 bytes in (an odd replay cut:
    pos[first:]). Returns (buffer, byte offset)."""
    if case["tensors"]["POS"][3]:
        return Buf.of(pos), 0
    b = Buf((len(pos) + 2) * 8)
    b.upload(pos, 8)
    return b, 8


def t_rope_heads():
    for ci, c in enumerate(LAUNCHES["_rope_heads"]):
        k = consts_of(c)
        Hh, HD, rd, inv, hb = k["H"], k["HD"], k["RD"], bool(k["INV"]), k["HB"]
        derived = ci >= NREC["_rope_heads"]
        for rows in (1, 17, 300) if derived else (c["tensors"]["X"][1][0],):
            check_rope(c, k, ci, derived, rows, Hh, HD, rd, inv, hb)


def check_rope(c, k, ci, derived, rows, Hh, HD, rd, inv, hb):
    """One _rope_heads launch of `rows` rows against the exact rotation, POS placed as the case says."""
    x = C.rand_bf16(rng, (rows, Hh, HD), 2.0)
    pos = rng.integers(0, 60000, rows).astype(np.int64)
    xb = Buf.of(x)
    pb, po = pos_buf(c, pos)
    tag = f"{'derived' if derived else 'rec'}{ci} H={Hh} HD={HD} inv={inv} rows={rows} POS aligned={po == 0}"
    args = dict(X=P(xb, "*bf16"), COS=P(cos_b, "*fp32"), SIN=P(sin_b, "*fp32"), POS=P(pb, "*i64", po))
    if derived:
        if not C.run_exact("_rope_heads", (rows, Hh // hb, 1), args, k, tag):
            return
    else:
        C.KIT.run("_rope_heads", (rows, Hh // hb, 1), args, k)
    got = C.from_bf16_bits(xb.download((rows, Hh, HD), np.uint16))
    xs = C.from_bf16_bits(x)
    exact = xs.copy()
    exact[..., -rd:] = R.rope_pairs(xs[..., -rd:], COS[pos][:, None, :].astype(np.float64),
                                    SIN[pos][:, None, :].astype(np.float64), inv)
    # numerics.json rope, hard: |dy| <= |x_pair|_2 (4u32 p f + 8u32) + u_out |y|; the reference rotates by the same
    # fp32 table the kernel reads (no table error), so |x_pair|_2 8u32 plus the output's rounding; statistical: nrmse
    # <= 4e-3 for a bf16 output. (One bf16 ulp is stricter where the rotation cancels: a result far smaller than
    # |x_pair| -- seen at 300 rows, 17 of 1.2M elements, the same bits with POS aligned or not.)
    pair = np.sqrt((xs[..., -rd:].reshape(rows, Hh, rd // 2, 2) ** 2).sum(-1))
    bnd = np.zeros_like(xs)
    bnd[..., -rd:] = np.repeat(pair, 2, axis=-1) * 8 * 2.0 ** -24
    m = C.cmp_rounded(got, exact, R.bf16(exact), C.ulp_bf16, bound=bnd)
    m["untouched_exact"] = bool(np.array_equal(got[..., :-rd], xs[..., :-rd]))
    m["ok"] = m["ok"] and m["untouched_exact"] and m["nl2"] <= 4e-3
    C.record(kernel="_rope_heads", case=tag, **m)




# ---- _embed_init, _tap, _collapse, _collapse_norm --------------------------------------------------------------------
def t_small():
    D, HC = 5120, 4
    V = 4096
    emb = C.rand_bf16(rng, (V, D))
    eb = Buf.of(emb)
    for rows in (1, 16):
        ids = rng.integers(0, V, rows).astype(np.int64)
        h, pre = Buf(rows * HC * D * 2), Buf(rows * HC * 4)
        k = consts_of(LAUNCHES["_embed_init"][0])
        C.KIT.run("_embed_init", (rows, D // 1024, 1), dict(EMB=P(eb, "*bf16"), IDS=P(Buf.of(ids), "*i64"),
                                                         H=P(h, "*bf16"), PRE=P(pre, "*fp32")), k)
        gh = h.download((rows, HC, D), np.uint16)
        gp = pre.download((rows, HC), np.float32)
        ok = np.array_equal(gh, np.repeat(emb[ids][:, None, :], HC, 1)) and np.array_equal(
            gp, np.tile(np.array([1, 0, 0, 0], np.float32), (rows, 1)))
        C.record(kernel="_embed_init", case=f"rows={rows}", ok=bool(ok), exact=True)

    for ci, c in enumerate(LAUNCHES["_tap"]):
        k = consts_of(c)
        for rows in (1, 16):
            x = C.rand_bf16(rng, (rows, HC, D))
            os_ = c["scalars"]["os_"]
            out = Buf(rows * os_ * 2)
            out.fill(0)
            C.KIT.run("_tap", (rows, D // 1024, 1), dict(H=P(Buf.of(x), "*bf16"), OUT=P(out, "*bf16"), os_=os_), k)
            got = C.from_bf16_bits(out.download((rows, os_), np.uint16))[:, :D]
            exact = C.from_bf16_bits(x).mean(1)
            m = C.cmp_rounded(got, exact, R.bf16(exact), C.ulp_bf16)
            C.record(kernel="_tap", case=f"rows={rows}", **m)

    for ci, c in enumerate(LAUNCHES["_collapse"]):
        k = consts_of(c)
        rows = c["tensors"]["X"][1][0]
        x = C.rand_bf16(rng, (rows, HC, D))
        pre = rng.random((rows, HC)).astype(np.float32) + np.float32(1e-6)
        out = Buf(rows * D * 2)
        C.KIT.run("_collapse", (rows, D // 1024, 1), dict(X=P(Buf.of(x), "*bf16"), PRE=P(Buf.of(pre), "*fp32"),
                                                       OUT=P(out, "*bf16")), k)
        got = C.from_bf16_bits(out.download((rows, D), np.uint16))
        terms = pre.astype(np.float64)[:, :, None] * C.from_bf16_bits(x)
        exact = terms.sum(1)
        m = C.cmp_rounded(got, exact, R.bf16(exact), C.ulp_bf16, bound=C.gamma(4) * np.abs(terms).sum(1))
        C.record(kernel="_collapse", case=f"rec{ci} rows={rows}", **m)

    for ci, c in enumerate(LAUNCHES["_collapse_norm"]):
        k = consts_of(c)
        rows = c["tensors"]["X"][1][0]
        x = C.rand_bf16(rng, (rows, HC, D))
        pre = rng.random((rows, HC)).astype(np.float32)
        w = C.rand_bf16(rng, (D,), 0.5)
        out = Buf(rows * D * 2)
        C.KIT.run("_collapse_norm", (rows, 1, 1), dict(X=P(Buf.of(x), "*bf16"), PRE=P(Buf.of(pre), "*fp32"),
                                                     NW=P(Buf.of(w), "*bf16"), OUT=P(out, "*bf16"), eps=EPS), k)
        got = C.from_bf16_bits(out.download((rows, D), np.uint16))
        terms = pre.astype(np.float64)[:, :, None] * C.from_bf16_bits(x)
        cx = terms.sum(1)
        cc = R.bf16(cx)
        rinv = 1 / np.sqrt(np.mean(cc * cc, -1, keepdims=True) + EPS)
        ws = C.from_bf16_bits(w)
        exact = ws * cc * rinv
        # c's own bound: its fp32 error, then possibly the other bf16 neighbour; the norm's fp32 error ~ (D + 8) u
        dc = C.gamma(4) * np.abs(terms).sum(1) + C.ulp_bf16(np.abs(cx) + C.gamma(4) * np.abs(terms).sum(1))
        bound = np.abs(ws) * rinv * dc + C.gamma(cx.shape[1] + 8) * np.abs(exact)
        m = C.cmp_rounded(got, exact, R.bf16(exact), C.ulp_bf16, bound=bound)
        C.record(kernel="_collapse_norm", case=f"rec{ci} rows={rows}", **m)


# ---- _engram_gate -----------------------------------------------------------------------------------------------
def t_engram_gate():
    D, HC = 5120, 4
    for ci, c in enumerate(LAUNCHES["_engram_gate"]):
        k = consts_of(c)
        rows = c["tensors"]["H"][1][0]
        h = C.rand_bf16(rng, (rows, HC, D))
        kv = C.rand_bf16(rng, (rows, (HC + 1) * D), 0.7)
        qk = (rng.standard_normal((HC, D)) * 0.3).astype(np.float32)
        out = Buf(rows * HC * D * 2)
        C.KIT.run("_engram_gate", (rows, HC, 1), dict(H=P(Buf.of(h), "*bf16"), KV=P(Buf.of(kv), "*bf16"),
                                                    QK=P(Buf.of(qk), "*fp32"), OUT=P(out, "*bf16"), eps=EPS), k)
        got = C.from_bf16_bits(out.download((rows, HC, D), np.uint16))
        hs = C.from_bf16_bits(h)
        kvs = C.from_bf16_bits(kv)
        key = kvs[:, :HC * D].reshape(rows, HC, D)
        val = kvs[:, HC * D:]
        rstd = 1 / np.sqrt((hs * hs).mean(-1) + EPS) / np.sqrt((key * key).mean(-1) + EPS)
        prod = hs * qk.astype(np.float64) * key
        dot = prod.sum(-1) * rstd / np.sqrt(D)
        g = lambda d: 1 / (1 + np.exp(-np.copysign(np.sqrt(np.maximum(np.abs(d), 1e-6)), d)))
        gate = g(dot)
        # the dot's fp32 error: the D-term sum (2 products a term) and the two rsqrt / scale factors
        ddot = (C.gamma(D + 2) * np.abs(prod).sum(-1)) * rstd / np.sqrt(D) + C.gamma(16) * np.abs(dot)
        dg = np.maximum(np.abs(g(dot + ddot) - gate), np.abs(g(dot - ddot) - gate)) + C.gamma(16) * gate
        exact = hs + gate[..., None] * val[:, None, :]
        bound = np.abs(val[:, None, :]) * dg[..., None] + C.gamma(2) * (np.abs(hs) + np.abs(gate[..., None] * val[:, None, :]))
        m = C.cmp_rounded(got, exact, R.bf16(exact), C.ulp_bf16, bound=bound)
        C.record(kernel="_engram_gate", case=f"rec{ci} rows={rows}", **m)


# ---- _hc_post ---------------------------------------------------------------------------------------------------
def t_hc_post():
    D, HC = 5120, 4
    for ci, c in enumerate(LAUNCHES["_hc_post"]):
        k = consts_of(c)
        world = k["WORLD"]
        rows = c["tensors"]["X"][1][0]
        g = (rng.standard_normal((world, rows, D)) * 0.5).astype(np.float32)
        x = C.rand_bf16(rng, (rows, HC, D))
        post = (2 * rng.random((rows, HC))).astype(np.float32)
        comb = rng.random((rows, HC, HC)).astype(np.float32)
        out = Buf(rows * HC * D * 2)
        C.KIT.run("_hc_post", (rows, D // 1024, 1), dict(G=P(Buf.of(g), "*fp32"), RS=rows * D, X=P(Buf.of(x), "*bf16"),
                                                      XOUT=P(out, "*bf16"), POST=P(Buf.of(post), "*fp32"),
                                                      COMB=P(Buf.of(comb), "*fp32")), k)
        got = C.from_bf16_bits(out.download((rows, HC, D), np.uint16))
        y = R.bf16(R.f32(g.astype(np.float64).sum(0)))      # the rank-ordered fp32 sum, then bf16
        exact = post.astype(np.float64)[:, :, None] * y[:, None, :] + np.einsum(
            "rjk,rjd->rkd", comb.astype(np.float64), C.from_bf16_bits(x))
        m = C.cmp_rounded(got, exact, R.bf16(exact), C.ulp_bf16)
        C.record(kernel="_hc_post", case=f"rec{ci} rows={rows} world={world}", **m)


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for t in (t_rmsnorm, t_rmsnorm_rot, t_kv_norm_rope, t_q_kv_norm, t_rope_heads, t_small, t_engram_gate, t_hc_post):
        if only and t.__name__ not in only:
            continue
        t()
    H.sync()
    sys.exit(C.done())
