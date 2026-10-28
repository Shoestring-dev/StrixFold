"""Sparse attention with a sink over the window (ring or linear) and the selected compressed entries (bf16 or packed
FP4 with E4M3 scales): _sparse_attn_part (split keys, or one pass for prompt chunks), _sparse_attn_merge,
_sparse_attn_merge_rot2 (merge + inverse RoPE + wo_a's input rotation); and the packed FP4 cache writes (_fp4_store)
-- every recorded variant against refs.py, and every derived _sparse_attn_merge_rot2 variant (variants.py: each window
size) at its own rows."""

import sys

import numpy as np

import common as C
import hiprt as H
import refs as R
from hiprt import Buf, P

LAUNCHES = C.launches()
rng = np.random.default_rng(2468)
C.log_open("t_attn")
HD, HB, BNK, WIN = 512, 16, 32, 128


def consts_of(case):
    return {k: (v if not isinstance(v, dict) else float(v["float"])) for k, v in case["consts"].items()}


def rope_tables(npos, rd, theta=10000.0):
    inv = 1.0 / theta ** (np.arange(0, rd, 2) / rd)
    ang = np.outer(np.arange(npos), inv)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


COS, SIN = rope_tables(70000, 64)
COS_B, SIN_B = Buf.of(COS), Buf.of(SIN)


def comp_pool(npool, packed):
    raw = rng.standard_normal((npool, HD)) * np.exp(rng.uniform(-1, 1, (npool, 1)))
    if not packed:
        b = C.bf16_bits(raw)
        return b, None, C.from_bf16_bits(b)
    deq, codes, sc = R.fp4_quant_rows(raw, 16, True)
    return R.fp4_pack(codes), sc, R.bf16(deq)


def attn_ref(q, keys_per_row, sink, scale):
    """o[r, h] = sum_j softmax_j k_j over the row's keys plus the sink (fp64); and a bound on the kernel's fp32 / bf16-p
    error: (2^-8 (p rounded to bf16 before the PV dot) + 4 max d s + gamma(n + 8)) * sum_j p_j |k_j| / l."""
    rows, Hh, _ = q.shape
    out = np.zeros((rows, Hh, HD))
    bnd = np.zeros((rows, Hh, HD))
    qs = C.from_bf16_bits(q)
    for r in range(rows):
        K = keys_per_row[r]                                    # [n, HD] fp64
        if K.shape[0] == 0:
            continue
        s = qs[r] @ K.T * scale                                # [H, n]
        ds = scale * C.gamma(HD) * (np.abs(qs[r]) @ np.abs(K).T) + C.gamma(2) * np.abs(s)
        m = np.maximum(s.max(1), sink)
        p = np.exp(s - m[:, None])
        lsum = p.sum(1) + np.exp(sink - m)
        out[r] = (p @ K) / lsum[:, None]
        pk = (p @ np.abs(K)) / lsum[:, None]
        bnd[r] = (2.0 ** -8 + 4 * ds.max(1)[:, None] + C.gamma(K.shape[0] + 8)) * pk
    return out, bnd


def build(case, rows=None):
    k = consts_of(case)
    t = case["tensors"]
    rows = rows or t["Q"][1][0]
    Hh = k["H"]
    ring, has_comp, packed, final, has_base = (bool(k[x]) for x in ("RING", "HAS_COMP", "PACKED", "FINAL", "HAS_BASE"))
    n_idx = case["scalars"].get("n_idx", k.get("n_idx", 0)) if has_comp else 0
    ring_size = case["scalars"]["ring_size"]
    q = C.rand_bf16(rng, (rows, Hh, HD), 0.35)
    sink = (rng.standard_normal(Hh) * 2).astype(np.float32)
    pos = rng.integers(0, 3000, rows).astype(np.int64)
    pos[0] = rng.integers(0, 60)                               # a window still filling
    if ring:
        nrings = 4 if has_base else 1
        wsrc = C.rand_bf16(rng, (ring_size * nrings, HD), 1.0)
        rbase = (rng.integers(0, nrings, rows) * ring_size).astype(np.int64) if has_base else None
        wlo = np.zeros(1, np.int64)
    else:
        pos = np.sort(rng.integers(0, 300, rows)).astype(np.int64) + 200
        wlo_v = int(pos.min()) - 40
        nlin = int(pos.max()) - wlo_v + 1
        wsrc = C.rand_bf16(rng, (nlin, HD), 1.0)
        wlo = np.array([wlo_v], np.int64)
        rbase = None
    npool = max(4 * n_idx, 64)
    codes, scales, deq = comp_pool(npool, packed) if has_comp else (wsrc, None, None)
    cbase = (rng.integers(0, npool // 2, rows)).astype(np.int64) if (has_base and has_comp) else None
    idx = None
    if has_comp:
        lim = npool // 2 if cbase is not None else npool
        idx = rng.integers(0, lim, (rows, n_idx)).astype(np.int64)
        idx[rng.random((rows, n_idx)) < 0.2] = -1
    return dict(k=k, rows=rows, Hh=Hh, ring=ring, has_comp=has_comp, packed=packed, final=final, has_base=has_base,
                n_idx=n_idx, ring_size=ring_size, q=q, sink=sink, pos=pos, wsrc=wsrc, rbase=rbase, wlo=wlo,
                codes=codes, scales=scales, deq=deq, cbase=cbase, idx=idx, scale=case["scalars"]["scale"]["float"])


def keys_of(d):
    """Each row's attended keys (fp64): the window positions p-127..p that exist, then the valid picks."""
    ws = C.from_bf16_bits(d["wsrc"])
    out = []
    for r in range(d["rows"]):
        p = int(d["pos"][r])
        wp = np.arange(p - WIN + 1, p + 1)
        wp = wp[wp >= 0]
        if d["ring"]:
            slots = wp % d["ring_size"] + (d["rbase"][r] if d["rbase"] is not None else 0)
        else:
            slots = wp - d["wlo"][0]
            slots = slots[slots >= 0]
        ks = [ws[slots]]
        if d["has_comp"]:
            ii = d["idx"][r]
            ii = ii[ii >= 0] + (d["cbase"][r] if d["cbase"] is not None else 0)
            ks.append(d["deq"][ii])
        out.append(np.concatenate(ks, 0))
    return out


def run_part(d, case, parts):
    k = d["k"]
    rows, Hh = d["rows"], d["Hh"]
    sp = k["SPLITS"]
    out = Buf(rows * Hh * HD * 2)
    pos_b = Buf.of(d["pos"])
    if d["final"]:
        pm = pl = po = P(out, "*bf16")
    else:
        pm, pl, po = (P(parts[0], "*fp32"), P(parts[1], "*fp32"), P(parts[2], "*fp32"))
    ct = "*u8" if d["packed"] else "*bf16"
    wsrc_b = Buf.of(d["wsrc"])
    codes_b = Buf.of(d["codes"]) if d["has_comp"] else wsrc_b
    n_idx = d["n_idx"]
    idx = d["idx"]
    if d["final"] and d["has_comp"] and n_idx % 16:
        idx = np.concatenate([idx, -np.ones((rows, 16 - n_idx % 16), np.int64)], 1)
        n_idx = idx.shape[1]
    args = dict(Q=P(Buf.of(d["q"]), "*bf16"), WSRC=P(wsrc_b, "*bf16"), WLO=P(Buf.of(d["wlo"]), "*i64"),
                COMP=P(codes_b, ct), CSC=P(Buf.of(d["scales"]), "*u8") if d["packed"] else P(wsrc_b, "*bf16"),
                IDX=P(Buf.of(idx), "*i64") if d["has_comp"] else P(pos_b, "*i64"), POS=P(pos_b, "*i64"),
                PM=pm, PL=pl, PO=po, SINK=P(Buf.of(d["sink"]), "*fp32"), OUT=P(out, "*bf16"), scale=d["scale"],
                RBASE=P(Buf.of(d["rbase"]), "*i64") if d["rbase"] is not None else P(pos_b, "*i64"),
                CBASE=P(Buf.of(d["cbase"] if d["cbase"] is not None else d["rbase"]), "*i64") if d["has_base"]
                else P(pos_b, "*i64"),
                ring_size=d["ring_size"], n_idx=n_idx)
    kk = dict(k)
    picks = (n_idx + BNK - 1) // BNK if d["has_comp"] else 0
    if d["final"] and picks:
        picks = 1 << (picks - 1).bit_length()
    kk["NBLK"] = WIN // BNK + picks
    if "n_idx" in kk:
        kk.pop("n_idx")
    C.KIT.run("_sparse_attn_part", (rows, Hh // HB, sp), args, kk)
    return out


def t_attention():
    for ci, c in enumerate(LAUNCHES["_sparse_attn_part"]):
        k = consts_of(c)
        rows = min(c["tensors"]["Q"][1][0], 40)
        d = build(c, rows)
        Hh, sp = d["Hh"], k["SPLITS"]
        parts = (Buf(rows * Hh * sp * 4), Buf(rows * Hh * sp * 4), Buf(rows * Hh * sp * HD * 4))
        out = run_part(d, c, parts)
        tag = (f"rec{ci} rows={rows} ring={d['ring']} comp={d['has_comp']} packed={d['packed']} base={d['has_base']} "
               f"final={d['final']} n_idx={d['n_idx']}")
        if not d["final"]:
            mk = consts_of(LAUNCHES["_sparse_attn_merge"][0])
            C.KIT.run("_sparse_attn_merge", (rows, Hh // HB, 1), dict(PM=P(parts[0], "*fp32"), PL=P(parts[1], "*fp32"),
                      PO=P(parts[2], "*fp32"), SINK=P(Buf.of(d["sink"]), "*fp32"), OUT=P(out, "*bf16")), mk)
            tag += " + merge"
        got = C.from_bf16_bits(out.download((rows, Hh, HD), np.uint16))
        keys = keys_of(d)
        ref, bnd = attn_ref(d["q"], keys, d["sink"].astype(np.float64), d["scale"])
        # contract: every element within the bound (p is rounded to bf16 before the PV dot, as in DeepSeek's kernel,
        # so ~40% of outputs legitimately differ from the correctly rounded value: no not-nearest limit) and the
        # normalised L2 error at the level of those roundings (bf16 output ~1.6e-3 rms, bf16 p ~1.2e-3)
        m = C.cmp_rounded(got, ref, R.bf16(ref), C.ulp_bf16, bound=bnd, max_frac=1.0)
        m["ok"] = m["ok"] and m["nl2"] <= 4e-3
        C.record(kernel="_sparse_attn_part", case=tag, **m)
        # negative control: the same check against a reference missing each row's oldest window key must fail
        drop = [k_[1:] if k_.shape[0] > 1 else k_ for k_ in keys]
        ref2, bnd2 = attn_ref(d["q"], drop, d["sink"].astype(np.float64), d["scale"])
        m2 = C.cmp_rounded(got, ref2, R.bf16(ref2), C.ulp_bf16, bound=bnd2, max_frac=1.0)
        caught = not (m2["ok"] and m2["nl2"] <= 4e-3)
        C.record(kernel="_sparse_attn_part", case=tag + " negative control (one key dropped)", ok=caught,
                 nl2=m2["nl2"], max_err_over_allowed=m2.get("max_err_over_allowed"))


def t_merge_rot2():
    """_sparse_attn_merge_rot2 at every window size R = 1 .. 16 with POS 16-byte aligned (round inputs since 05832c5),
    and at 1 row with POS unaligned (the recorded launch), each on its own kit variant (exact selection, as the
    engine's strict mode; derived variants: variants.py): the split partials of a verify round's _sparse_attn_part case
    (packed FP4 picks) merged, inverse RoPE on the last 64 dims, wo_a's rotation."""
    base = LAUNCHES["_sparse_attn_part"][4]
    mk = consts_of(LAUNCHES["_sparse_attn_merge"][0])
    rk = {k: v for k, v in consts_of(LAUNCHES["_sparse_attn_merge_rot2"][0]).items() if k != "rows"}
    for rows, aligned in [(1, False)] + [(r, True) for r in range(1, 17)]:
        tag = f"rows={rows} POS aligned={aligned}"
        d = build(base, rows)
        Hh, sp = d["Hh"], d["k"]["SPLITS"]
        parts = (Buf(rows * Hh * sp * 4), Buf(rows * Hh * sp * 4), Buf(rows * Hh * sp * HD * 4))
        out = run_part(d, base, parts)
        C.KIT.run("_sparse_attn_merge", (rows, Hh // HB, 1), dict(PM=P(parts[0], "*fp32"), PL=P(parts[1], "*fp32"),
                  PO=P(parts[2], "*fp32"), SINK=P(Buf.of(d["sink"]), "*fp32"), OUT=P(out, "*bf16")), mk)
        got = C.from_bf16_bits(out.download((rows, Hh, HD), np.uint16))
        GH = rk["GH"]
        suh = np.sign(rng.standard_normal(Hh * HD)).astype(np.float16)
        pos_b = Buf((rows + 2) * 8)
        pos_b.upload(d["pos"], 0 if aligned else 8)
        out2, xh = Buf(rows * Hh * HD * 2), Buf((Hh // GH) * rows * GH * HD * 2)
        args = dict(PM=P(parts[0], "*fp32"), PL=P(parts[1], "*fp32"), PO=P(parts[2], "*fp32"),
                    SINK=P(Buf.of(d["sink"]), "*fp32"), OUT=P(out2, "*bf16"), COS=P(COS_B, "*fp32"),
                    SIN=P(SIN_B, "*fp32"), POS=P(pos_b, "*i64", 0 if aligned else 8), SUH=P(Buf.of(suh), "*fp16"),
                    XH=P(xh, "*fp16"), rows=rows)
        consts = rk
        try:
            var = C.KIT.find("_sparse_attn_merge_rot2", args, consts, exact_only=True)
        except KeyError as e:
            C.record(kernel="_sparse_attn_merge_rot2", case=tag + " exact variant in the kit", ok=False, error=str(e))
            continue
        C.KIT.run("_sparse_attn_merge_rot2", (rows, Hh, 1), args, consts, variant=var)
        g2 = C.from_bf16_bits(out2.download((rows, Hh, HD), np.uint16))
        # the merged bf16 row is _sparse_attn_merge's (graded in t_attention); the kernel ropes that row: exact
        # reference of that step
        ex = got.copy()
        ex[..., -64:] = R.rope_pairs(got[..., -64:], COS[d["pos"]][:, None, :].astype(np.float64),
                                     SIN[d["pos"]][:, None, :].astype(np.float64), inverse=True)
        m = C.cmp_rounded(g2, ex, R.bf16(ex), C.ulp_bf16)
        m["nope_dims_equal_merge"] = bool(np.array_equal(g2[..., :-64], got[..., :-64]))
        m["ok"] = m["ok"] and m["nope_dims_equal_merge"]
        C.record(kernel="_sparse_attn_merge_rot2", case=f"{tag} merge + inverse rope", **m)
        gx = xh.download((Hh // GH, rows, GH * HD), np.float16).astype(np.float64)
        want = np.zeros_like(gx)
        for h in range(Hh):
            v = g2[:, h, :] * suh[h * HD:(h + 1) * HD].astype(np.float64)
            want[h // GH, :, (h % GH) * HD:(h % GH + 1) * HD] = (v.reshape(rows, HD // 128, 128) @ R.H128.T
                                                               / np.sqrt(128)).reshape(rows, HD)
        m = C.cmp_rounded(gx, want, R.f16(want), C.ulp_f16)
        C.record(kernel="_sparse_attn_merge_rot2", case=f"{tag} wo_a rotation", **m)


def t_fp4_store():
    for ci, c in enumerate(LAUNCHES["_fp4_store"]):
        k = consts_of(c)
        D, blk, e4 = k["D"], k["BLOCK"], bool(k["E4M3"])
        for n in (1, 16):
            x = C.rand_bf16(rng, (n, D), 1.0)
            xs = C.from_bf16_bits(x)
            xs[0, :blk] = 0.0                                   # an all-zero block
            xs[0, blk:2 * blk] *= 1e-6
            x = C.bf16_bits(xs)
            ncache = 64
            rowsel = rng.permutation(ncache)[:n].astype(np.int64)
            codes0 = rng.integers(0, 256, (ncache, D // 2)).astype(np.uint8)
            sc0 = rng.integers(0, 256, (ncache, D // blk)).astype(np.uint8)
            cb, sb = Buf.of(codes0), Buf.of(sc0)
            C.KIT.run("_fp4_store", (n, 1, 1), dict(X=P(Buf.of(x), "*bf16"), xs=D, CODES=P(cb, "*u8"), SCALES=P(sb, "*u8"),
                      ROWS=P(Buf.of(rowsel), "*i64")), k)
            _, codes, sbytes = R.fp4_quant_rows(C.from_bf16_bits(x), blk, e4)
            wc, ws = codes0.copy(), sc0.copy()
            wc[rowsel] = R.fp4_pack(codes)
            ws[rowsel] = sbytes
            gc, gs = cb.download(codes0.shape, np.uint8), sb.download(sc0.shape, np.uint8)
            C.record(kernel="_fp4_store", case=f"rec{ci} D={D} block={blk} e4m3={e4} rows={n}",
                     ok=bool(np.array_equal(gc, wc) and np.array_equal(gs, ws)),
                     code_mismatch=int((gc != wc).sum()), scale_mismatch=int((gs != ws).sum()))


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for t in (t_attention, t_merge_rot2, t_fp4_store):
        if only and t.__name__ not in only:
            continue
        t()
    H.sync()
    sys.exit(C.done())
