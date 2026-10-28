"""mHC kernels: the mixes' partial dots (_hc_partial, _hc_mix_pf, _hc_mix_part, posted or not), the posted streams
(_hc_mix_pf / _hc_mix_part POSTED, _hc_post_only), the finish (_hc_finish, _hc_finish_s roles, _hc_sinkhorn) -- every
recorded variant, on the AOT kit, against refs.py; plus row invariance (a row's bits alone vs inside a window)."""

import sys

import numpy as np

import common as C
import hiprt as H
import refs as R
from hiprt import Buf, P

LAUNCHES = C.launches()
EPS, HC_EPS, ITERS = 1e-20, 1e-6, 20
D, HC, NB, WIDE = 5120, 4, 40, 20480
KB = WIDE // NB
rng = np.random.default_rng(4321)
C.log_open("t_hc")

FN = (rng.standard_normal((24, WIDE)) * 0.01).astype(np.float32)
FN_B = Buf.of(FN)
SCALE = np.array([0.8, 1.1, 0.6], np.float32)
BASE = (rng.standard_normal(24) * 0.5).astype(np.float32)
NW = C.rand_bf16(rng, (D,), 0.5)
SCALE_B, BASE_B, NW_B = Buf.of(SCALE), Buf.of(BASE), Buf.of(NW)


def consts_of(case):
    return {k: (v if not isinstance(v, dict) else float(v["float"])) for k, v in case["consts"].items()}


def streams(rows):
    return C.rand_bf16(rng, (rows, HC, D), 1.0)


def posted_inputs(rows, world):
    g = (rng.standard_normal((world, rows, D)) * 0.5).astype(np.float32)
    post = (2 * rng.random((rows, HC))).astype(np.float32)
    comb = rng.random((rows, HC, HC)).astype(np.float32)
    return g, post, comb


def post_ref(g, x, post, comb):
    """hc_post of the gathered partials: y = bf16(fp32 rank-ordered sum); out[k] = post[k] y + sum_j comb[j,k] x_j;
    returns (exact, bound) -- the fp32 error bound of the 5-term expression before its bf16 rounding."""
    y = R.bf16(R.f32(g.astype(np.float64).sum(0)))
    xs = C.from_bf16_bits(x)
    t1 = post.astype(np.float64)[:, :, None] * y[:, None, :]
    t2 = comb.astype(np.float64)[:, :, :, None] * xs[:, :, None, :]       # [r, j, k, d]
    exact = t1 + t2.sum(1)
    bound = C.gamma(5) * (np.abs(t1) + np.abs(t2).sum(1))
    return exact, bound


def check_part(name, case, part, xs_bf16, rows):
    """PART [rows, NB, 32]: block b's 24 dots of fn with the stream block and its sum of squares (slot 24)."""
    xf = C.from_bf16_bits(xs_bf16).reshape(rows, NB, KB)
    fb = FN.astype(np.float64).reshape(24, NB, KB)
    dots = np.einsum("rbk,mbk->rbm", xf, fb)
    adots = np.einsum("rbk,mbk->rbm", np.abs(xf), np.abs(fb))
    ss = (xf * xf).sum(-1)
    m1 = C.cmp_bound(part[:, :, :24], dots, C.gamma(KB) * adots)
    m2 = C.cmp_bound(part[:, :, 24], ss, C.gamma(KB) * ss)
    C.record(kernel=name, case=case + " dots", **m1)
    C.record(kernel=name, case=case + " sumsq", **m2)


def run_mix_part(case, x, rows, rb, posted=None, xo=None):
    k = consts_of(case)
    k["RB"] = rb
    xb = Buf.of(x)
    part = Buf(rows * NB * 32 * 4)
    part.fill(0)
    if posted is None:
        args = dict(X=P(xb, "*bf16"), XO=P(xb, "*bf16"), G=P(xb, "*bf16"), RS=0, POST=P(part, "*fp32"),
                    COMB=P(part, "*fp32"), FN=P(FN_B, "*fp32"), PART=P(part, "*fp32"), rows=rows)
        xob = xb
    else:
        g, post, comb = posted
        xob = Buf(rows * HC * D * 2)
        args = dict(X=P(xb, "*bf16"), XO=P(xob, "*bf16"), G=P(Buf.of(g), "*fp32"), RS=rows * D,
                    POST=P(Buf.of(post), "*fp32"), COMB=P(Buf.of(comb), "*fp32"), FN=P(FN_B, "*fp32"),
                    PART=P(part, "*fp32"), rows=rows)
    C.KIT.run("_hc_mix_part", ((rows + rb - 1) // rb, NB, 3), args, k)
    return part.download((rows, NB, 32), np.float32).astype(np.float64), xob.download((rows, HC, D), np.uint16)


def t_partial_and_pf():
    c = LAUNCHES["_hc_partial"][0]
    for rows in (1, 7):
        x = streams(rows)
        part = Buf(rows * NB * 32 * 4)
        C.KIT.run("_hc_partial", (rows, NB, 1), dict(X=P(Buf.of(x), "*bf16"), FN=P(FN_B, "*fp32"),
                                                     PART=P(part, "*fp32")), consts_of(c))
        check_part("_hc_partial", f"rows={rows}", part.download((rows, NB, 32), np.float32).astype(np.float64),
                   x, rows)
    for ci, c in enumerate(LAUNCHES["_hc_mix_pf"]):
        k = consts_of(c)
        world = k["WORLD"]
        for rows in (1, 17, 512):
            rb = max(1, min(8, rows))
            if (rows == 1) != (ci == 0):
                continue                                        # rec0 is the rows=1 variant (rows folded), rec1 rows>1
            k2 = dict(k)
            k2["RB"] = rb
            x = streams(rows)
            for posted in (True, False):
                if posted != bool(k["POSTED"]):
                    pass
                part = Buf(rows * NB * 32 * 4)
                if posted:
                    g, post, comb = posted_inputs(rows, world)
                    xob = Buf(rows * HC * D * 2)
                    args = dict(X=P(Buf.of(x), "*bf16"), XO=P(xob, "*bf16"), G=P(Buf.of(g), "*fp32"), RS=rows * D,
                                POST=P(Buf.of(post), "*fp32"), COMB=P(Buf.of(comb), "*fp32"), FN=P(FN_B, "*fp32"),
                                PART=P(part, "*fp32"), rows=rows)
                    C.KIT.run("_hc_mix_pf", (((rows + rb - 1) // rb) * NB, 1, 1), args, k2)
                    xo = xob.download((rows, HC, D), np.uint16)
                    exact, bound = post_ref(g, x, post, comb)
                    m = C.cmp_rounded(C.from_bf16_bits(xo), exact, R.bf16(exact), C.ulp_bf16, bound=bound)
                    C.record(kernel="_hc_mix_pf", case=f"rec{ci} rows={rows} posted streams", **m)
                    check_part("_hc_mix_pf", f"rec{ci} rows={rows} posted", part.download(
                        (rows, NB, 32), np.float32).astype(np.float64), xo, rows)
                else:
                    kk = C.extra_variant("_hc_mix_pf", c, POSTED=False, WORLD=1, RB=rb)
                    xb = Buf.of(x)
                    args = dict(X=P(xb, "*bf16"), XO=P(xb, "*bf16"), G=P(xb, "*bf16"), RS=0, POST=P(part, "*fp32"),
                                COMB=P(part, "*fp32"), FN=P(FN_B, "*fp32"), PART=P(part, "*fp32"), rows=rows)
                    if rows == 1:
                        args.pop("rows")
                    try:
                        C.run_variant(kk, (((rows + rb - 1) // rb) * NB, 1, 1), args)
                    except KeyError as e:
                        C.record(kernel="_hc_mix_pf", case=f"rows={rows} plain", ok=False, error=str(e)[:200])
                        continue
                    check_part("_hc_mix_pf", f"rows={rows} plain (POSTED=0 extra variant)",
                               part.download((rows, NB, 32), np.float32).astype(np.float64), x, rows)


def t_mix_part():
    """Every recorded _hc_mix_part variant; then row invariance: each row's PART / XO bits equal its 1-row launch."""
    seen = set()
    for ci, c in enumerate(LAUNCHES["_hc_mix_part"]):
        k = consts_of(c)
        rows = c["scalars"].get("rows", k.get("rows", 1))
        rb, posted, world = k["RB"], bool(k["POSTED"]), k["WORLD"]
        x = streams(rows)
        pin = posted_inputs(rows, world) if posted else None
        part, xo = run_mix_part(c, x, rows, rb, pin)
        tag = f"rec{ci} rows={rows} RB={rb} posted={posted} pdl={k['PDL']}"
        if posted:
            exact, bound = post_ref(pin[0], x, pin[1], pin[2])
            m = C.cmp_rounded(C.from_bf16_bits(xo), exact, R.bf16(exact), C.ulp_bf16, bound=bound)
            C.record(kernel="_hc_mix_part", case=tag + " posted streams", **m)
        check_part("_hc_mix_part", tag, part, xo if posted else x, rows)
        # row invariance: each row alone (rows=1 variant of the same POSTED/WORLD/PDL) gives the same bits
        key = (posted, world, k["PDL"], rows)
        if key in seen or rows == 1:
            continue
        seen.add(key)
        one = next((cc for cc in LAUNCHES["_hc_mix_part"] if consts_of(cc).get("rows", cc["scalars"].get("rows")) == 1
                    and bool(cc["consts"]["POSTED"]) == posted and cc["consts"]["PDL"] == k["PDL"]), None)
        if one is None:
            one = c
        same_part = same_xo = True
        for r in range(rows):
            p1 = (pin[0][:, r:r + 1].copy(), pin[1][r:r + 1], pin[2][r:r + 1]) if posted else None
            if one is c:
                kk = C.extra_variant("_hc_mix_part", c, RB=1)
                part1, xo1 = run_mix_part_variant(kk, x[r:r + 1], p1)
            else:
                part1, xo1 = run_mix_part(one, x[r:r + 1], 1, 1, p1)
            same_part &= bool(np.array_equal(part1[0], part[r]))
            same_xo &= bool(np.array_equal(xo1[0], xo[r]))
        C.record(kernel="_hc_mix_part", case=f"row invariance rows={rows} RB={rb} posted={posted}",
                 ok=same_part and same_xo, part_bits_equal=same_part, streams_bits_equal=same_xo)


def run_mix_part_variant(kk, x, posted):
    rows = 1
    xb = Buf.of(x)
    part = Buf(rows * NB * 32 * 4)
    if posted is None:
        args = dict(X=P(xb, "*bf16"), XO=P(xb, "*bf16"), G=P(xb, "*bf16"), RS=0, POST=P(part, "*fp32"),
                    COMB=P(part, "*fp32"), FN=P(FN_B, "*fp32"), PART=P(part, "*fp32"), rows=1)
        xob = xb
    else:
        g, post, comb = posted
        xob = Buf(HC * D * 2)
        args = dict(X=P(xb, "*bf16"), XO=P(xob, "*bf16"), G=P(Buf.of(g), "*fp32"), RS=D, POST=P(Buf.of(post), "*fp32"),
                    COMB=P(Buf.of(comb), "*fp32"), FN=P(FN_B, "*fp32"), PART=P(part, "*fp32"), rows=1)
    C.run_variant(kk, (1, NB, 3), args)
    return part.download((1, NB, 32), np.float32).astype(np.float64), xob.download((1, HC, D), np.uint16)


# ---- the finish ---------------------------------------------------------------------------------------------------
def sinkhorn_ref(part):
    """From the partial sums: mix (fp64 sums), the scale, pre/post/comb; and a propagated tolerance."""
    p = part.astype(np.float64)
    mixraw = p[:, :, :24].sum(1)
    ss = p[:, :, 24].sum(1)
    rs = 1.0 / np.sqrt(ss / (4 * D) + EPS)
    mix = mixraw * rs[:, None]
    pre, post, comb = R.hc_split_sinkhorn(mix, SCALE, BASE, HC, ITERS, HC_EPS)
    # the mixes' fp32 error: the 40-term sums and the scale (sqrt, divisions)
    dmix = (C.gamma(NB) * np.abs(p[:, :, :24]).sum(1) * rs[:, None] + C.gamma(NB + 8) * np.abs(mix))
    devs = [np.zeros_like(pre), np.zeros_like(post), np.zeros_like(comb)]
    prng = np.random.default_rng(7)
    for t in range(16):
        sgn = prng.choice([-1.0, 1.0], size=mix.shape) if t > 1 else np.full(mix.shape, 1.0 if t == 0 else -1.0)
        a, b, cc = R.hc_split_sinkhorn(mix + sgn * dmix, SCALE, BASE, HC, ITERS, HC_EPS)
        devs = [np.maximum(devs[0], np.abs(a - pre)), np.maximum(devs[1], np.abs(b - post)),
                np.maximum(devs[2], np.abs(cc - comb))]
    # the fp32 arithmetic of the finish itself: exp / sigmoid (a few ulp), 20 Sinkhorn rounds of 4-term sums and
    # divisions (contractive normalisations: errors do not grow), budgeted 4 * (8 * ITERS) u relative
    tol = [2 * devs[0] + C.gamma(32) * np.abs(pre), 2 * devs[1] + C.gamma(32) * np.abs(post),
           2 * devs[2] + C.gamma(32 * ITERS) * np.abs(comb)]
    return (pre, post, comb), tol


def collapse_norm_ref(x, pre_in):
    terms = pre_in.astype(np.float64)[:, :, None] * C.from_bf16_bits(x)
    cx = terms.sum(1)
    cc = R.bf16(cx)
    rinv = 1 / np.sqrt(np.mean(cc * cc, -1, keepdims=True) + EPS)
    ws = C.from_bf16_bits(NW)
    exact = ws * cc * rinv
    dc = C.gamma(4) * np.abs(terms).sum(1) + C.ulp_bf16(np.abs(cx) + C.gamma(4) * np.abs(terms).sum(1))
    bound = np.abs(ws) * rinv * dc + C.gamma(D + 8) * np.abs(exact)
    return exact, bound


def finish_inputs(rows):
    x = streams(rows)
    # PART from the real dots of x (so the scale is consistent), as the mixes kernel would produce them
    xf = C.from_bf16_bits(x).reshape(rows, NB, KB)
    part = np.zeros((rows, NB, 32), np.float32)
    part[:, :, :24] = np.einsum("rbk,mbk->rbm", xf, FN.astype(np.float64).reshape(24, NB, KB)).astype(np.float32)
    part[:, :, 24] = (xf * xf).sum(-1).astype(np.float32)
    pre_in = (rng.random((rows, HC)) + 1e-6).astype(np.float32)
    return x, part, pre_in


def check_finish(name, tag, outs, x, part, pre_in, sink=True, norm=True):
    (pre, post, comb), tol = sinkhorn_ref(part)
    if sink:
        for nm, got, ref, t in zip(("pre", "post", "comb"), outs[:3], (pre, post, comb), tol):
            m = C.cmp_bound(got.reshape(ref.shape), ref, t)
            C.record(kernel=name, case=f"{tag} {nm}", **m)
    if norm:
        exact, bound = collapse_norm_ref(x, pre_in)
        m = C.cmp_rounded(C.from_bf16_bits(outs[3]), exact, R.bf16(exact), C.ulp_bf16, bound=bound)
        C.record(kernel=name, case=f"{tag} collapse+norm", **m)


def finish_bufs(rows, x, part, pre_in):
    b = dict(X=Buf.of(x), PART=Buf.of(part), PRE_IN=Buf.of(pre_in), OUT=Buf(rows * D * 2), PRE_OUT=Buf(rows * HC * 4),
             POST=Buf(rows * HC * 4), COMB=Buf(rows * HC * HC * 4))
    for v in ("PRE_OUT", "POST", "COMB", "OUT"):
        b[v].fill(0xFF)
    return b


def finish_args(b):
    return dict(X=P(b["X"], "*bf16"), PART=P(b["PART"], "*fp32"), BASE=P(BASE_B, "*fp32"), SCALE=P(SCALE_B, "*fp32"),
                PRE_IN=P(b["PRE_IN"], "*fp32"), NW=P(NW_B, "*bf16"), OUT=P(b["OUT"], "*bf16"),
                PRE_OUT=P(b["PRE_OUT"], "*fp32"), POST=P(b["POST"], "*fp32"), COMB=P(b["COMB"], "*fp32"),
                eps=EPS, hc_eps=HC_EPS)


def read_finish(b, rows):
    return (b["PRE_OUT"].download((rows, HC), np.float32).astype(np.float64),
            b["POST"].download((rows, HC), np.float32).astype(np.float64),
            b["COMB"].download((rows, HC, HC), np.float32).astype(np.float64),
            b["OUT"].download((rows, D), np.uint16))


def t_finish():
    c = LAUNCHES["_hc_finish"][0]
    for rows in (1, 9):
        x, part, pre_in = finish_inputs(rows)
        b = finish_bufs(rows, x, part, pre_in)
        C.KIT.run("_hc_finish", (rows, 1, 1), finish_args(b), consts_of(c))
        check_finish("_hc_finish", f"rows={rows}", read_finish(b, rows), x, part, pre_in)
    c = LAUNCHES["_hc_sinkhorn"][0]
    for rows in (1, 16):
        x, part, pre_in = finish_inputs(rows)
        b = finish_bufs(rows, x, part, pre_in)
        a = finish_args(b)
        for v in ("X", "PRE_IN", "NW", "OUT"):
            a.pop(v)
        C.KIT.run("_hc_sinkhorn", (rows, 1, 1), a, consts_of(c))
        check_finish("_hc_sinkhorn", f"rows={rows}", read_finish(b, rows), x, part, pre_in, norm=False)
    # _hc_finish_s: SINK=True (roles 0 and 1) and SINK=False (role 1 only); S0..H3 all OUT (NROT 0)
    ref_bits = {}
    for ci, c in enumerate(LAUNCHES["_hc_finish_s"]):
        k = consts_of(c)
        sink = bool(k["SINK"])
        for rows in sorted({1, c["tensors"]["X"][1][0]}):
            x, part, pre_in = finish_inputs(rows)
            b = finish_bufs(rows, x, part, pre_in)
            a = finish_args(b)
            for i in range(4):
                a[f"S{i}"] = a[f"H{i}"] = P(b["OUT"], "*bf16")
            C.KIT.run("_hc_finish_s", (rows, 2 if sink else 1, 1), a, k)
            outs = read_finish(b, rows)
            check_finish("_hc_finish_s", f"rec{ci} rows={rows} sink={sink}", outs, x, part, pre_in, sink=sink)
            if not sink:
                untouched = bool(np.all(b["COMB"].download((rows * 16,), np.uint32) == 0xFFFFFFFF))
                C.record(kernel="_hc_finish_s", case=f"rec{ci} rows={rows} SINK=0 leaves pre/post/comb", ok=untouched)


def t_post_only():
    for ci, c in enumerate(LAUNCHES["_hc_post_only"]):
        k = consts_of(c)
        world = k["WORLD"]
        for rows in (1, 16):
            x = streams(rows)
            g, post, comb = posted_inputs(rows, world)
            xob = Buf(rows * HC * D * 2)
            C.KIT.run("_hc_post_only", (rows, NB, 1), dict(X=P(Buf.of(x), "*bf16"), XO=P(xob, "*bf16"),
                      G=P(Buf.of(g), "*fp32"), RS=rows * D, POST=P(Buf.of(post), "*fp32"),
                      COMB=P(Buf.of(comb), "*fp32")), k)
            exact, bound = post_ref(g, x, post, comb)
            got = C.from_bf16_bits(xob.download((rows, HC, D), np.uint16))
            m = C.cmp_rounded(got, exact, R.bf16(exact), C.ulp_bf16, bound=bound)
            C.record(kernel="_hc_post_only", case=f"rec{ci} rows={rows}", **m)


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for t in (t_partial_and_pf, t_mix_part, t_finish, t_post_only):
        if only and t.__name__ not in only:
            continue
        t()
    H.sync()
    sys.exit(C.done())
