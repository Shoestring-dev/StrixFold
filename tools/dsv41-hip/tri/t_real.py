"""The auxiliary kernels on the checkpoint's own tensors (LibertAI 2.9 bpw @ c178a012, read straight from the
safetensors by header offset): layer 2 (KV/index source, compressor), layer 14 (Engram + KV source), layer 20 (the
candidate source), mtp.2 (the drafter's last stage, Markov head). Activations are synthetic (no reference capture
exists yet); the weights, norms, mHC mixing tensors, router weights and biases, sinks and the Markov head are real.
Same fp64 references and contracts as the per-kernel suites."""

import json
import sys
from pathlib import Path

import numpy as np

import common as C
import hiprt as H
import refs as R
from hiprt import Buf, P
import t_route as TR                     # its route_check (imported before this file's log opens)

PACK = Path(__import__("os").environ.get("TRI_PACK", Path.home() / "models/tf-port/libertai-2.9bpw"))


def pack_index():
    """Every tensor's (file, dtype, shape, data offsets) from the safetensors headers (cached in upstream/)."""
    path = C.ROOT / "upstream" / "pack_index_c178a012.json"
    if not path.exists():
        idx = {}
        for f in sorted(PACK.glob("model-*.safetensors")):
            with open(f, "rb") as fh:
                h = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
            idx.update({k: [f.name, v["dtype"], v["shape"], v["data_offsets"]] for k, v in h.items() if k != "__metadata__"})
        path.write_text(json.dumps(idx))
    return json.loads(path.read_text())


IDX = pack_index()
L = C.launches()
rng = np.random.default_rng(2026)
C.log_open("t_real")
DT = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}
D, HC, NB, WIDE = 5120, 4, 40, 20480
EPS, HC_EPS = 1e-20, 1e-6


def tensor(name):
    f, dt, shape, (a, b) = IDX[name]
    with open(PACK / f, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        fh.seek(8 + n + a)
        raw = fh.read(b - a)
    return np.frombuffer(raw, dtype=DT[dt]).reshape(shape).copy()


def k_of(fn, pred):
    for c in L[fn]:
        k = {kk: (v if not isinstance(v, dict) else float(v["float"])) for kk, v in c["consts"].items()}
        if pred(c, k):
            return k
    raise KeyError(fn)


def streams(rows, scale=1.0):
    # residual streams of a mid-network magnitude: per-row scale varies, a few large channels
    x = rng.standard_normal((rows, HC, D)) * np.exp(rng.uniform(-1, 1, (rows, 1, 1))) * scale
    x[:, :, rng.integers(0, D, 8)] *= 20
    return C.bf16_bits(x)


def t_norms(layer):
    for nm, d in ((f"{layer}.attn_norm.weight", 5120), (f"{layer}.attn.q_norm.weight", 1280),
                  (f"{layer}.attn.kv_norm.weight", 512)):
        if nm not in IDX:
            continue
        w = tensor(nm)
        k = k_of("_rmsnorm", lambda c, k: k["D"] == d)
        for rows in (1, 6, 512):
            x = C.rand_bf16(rng, (rows, d), 3.0)
            out = Buf(rows * d * 2)
            C.KIT.run("_rmsnorm", (rows, 1, 1), dict(X=P(Buf.of(x), "*bf16"), xs=d, W=P(Buf.of(w), "*bf16"),
                      OUT=P(out, "*bf16"), os_=d, eps=EPS), k)
            xs, ws = C.from_bf16_bits(x), C.from_bf16_bits(w)
            ex = ws * xs / np.sqrt(np.mean(xs * xs, -1, keepdims=True) + EPS)
            m = C.cmp_rounded(C.from_bf16_bits(out.download((rows, d), np.uint16)), ex, R.bf16(ex), C.ulp_bf16,
                              bound=C.gamma(d + 8) * np.abs(ex))
            C.record(kernel="_rmsnorm", case=f"real {nm} rows={rows}", **m)


def t_mhc(layer, which="attn"):
    fn = tensor(f"{layer}.hc_{which}_fn")
    base = tensor(f"{layer}.hc_{which}_base")
    scale = tensor(f"{layer}.hc_{which}_scale")
    nw = tensor(f"{layer}.{which}_norm.weight")
    fnb, bb, sb, nwb = Buf.of(fn), Buf.of(base), Buf.of(scale), Buf.of(nw)
    kpart = k_of("_hc_partial", lambda c, k: True)
    kfin = k_of("_hc_finish", lambda c, k: True)
    for rows in (1, 6, 64):
        x = streams(rows)
        part = Buf(rows * NB * 32 * 4)
        pre_in = (rng.random((rows, HC)) + 1e-6).astype(np.float32)
        outs = dict(OUT=Buf(rows * D * 2), PRE_OUT=Buf(rows * 16), POST=Buf(rows * 16), COMB=Buf(rows * 64))
        xb = Buf.of(x)
        C.KIT.run("_hc_partial", (rows, NB, 1), dict(X=P(xb, "*bf16"), FN=P(fnb, "*fp32"), PART=P(part, "*fp32")), kpart)
        C.KIT.run("_hc_finish", (rows, 1, 1), dict(X=P(xb, "*bf16"), PART=P(part, "*fp32"), BASE=P(bb, "*fp32"),
                  SCALE=P(sb, "*fp32"), PRE_IN=P(Buf.of(pre_in), "*fp32"), NW=P(nwb, "*bf16"), OUT=P(outs["OUT"], "*bf16"),
                  PRE_OUT=P(outs["PRE_OUT"], "*fp32"), POST=P(outs["POST"], "*fp32"), COMB=P(outs["COMB"], "*fp32"),
                  eps=EPS, hc_eps=HC_EPS), kfin)
        # end to end against the model's hc_mixes / hc_split_sinkhorn / hc_pre / attn_norm in fp64
        xs = C.from_bf16_bits(x)
        xf = xs.reshape(rows, -1)
        mix = (xf @ fn.astype(np.float64).T) / np.sqrt(np.mean(xf * xf, 1, keepdims=True) + EPS)
        pre, post, comb = R.hc_split_sinkhorn(mix, scale, base, HC, 20, HC_EPS)
        dmix = C.gamma(600) * (np.abs(xf) @ np.abs(fn.astype(np.float64)).T) / np.sqrt(np.mean(xf * xf, 1, keepdims=True) + EPS)
        devs = [0, 0, 0]
        for t in range(12):
            sg = rng.choice([-1.0, 1.0], size=mix.shape) if t > 1 else np.full(mix.shape, 1.0 - 2 * t)
            a, b_, cc = R.hc_split_sinkhorn(mix + sg * dmix, scale, base, HC, 20, HC_EPS)
            devs = [np.maximum(devs[0], np.abs(a - pre)), np.maximum(devs[1], np.abs(b_ - post)),
                    np.maximum(devs[2], np.abs(cc - comb))]
        for nm, buf, ref, dv, sh in (("pre", "PRE_OUT", pre, devs[0], (rows, HC)), ("post", "POST", post, devs[1], (rows, HC)),
                                     ("comb", "COMB", comb, devs[2], (rows, HC, HC))):
            got = outs[buf].download(sh, np.float32).astype(np.float64)
            m = C.cmp_bound(got, ref, 2 * dv + C.gamma(32 * 20) * np.abs(ref))
            C.record(kernel="mHC prompt (_hc_partial+_hc_finish)", case=f"real {layer}.hc_{which} rows={rows} {nm}", **m)
        terms = pre_in.astype(np.float64)[:, :, None] * xs
        cx = terms.sum(1)
        cc = R.bf16(cx)
        rinv = 1 / np.sqrt(np.mean(cc * cc, -1, keepdims=True) + EPS)
        ws = C.from_bf16_bits(nw)
        ex = ws * cc * rinv
        dc = C.gamma(4) * np.abs(terms).sum(1) + C.ulp_bf16(np.abs(cx) + C.gamma(4) * np.abs(terms).sum(1))
        m = C.cmp_rounded(C.from_bf16_bits(outs["OUT"].download((rows, D), np.uint16)), ex, R.bf16(ex), C.ulp_bf16,
                          bound=np.abs(ws) * rinv * dc + C.gamma(D + 8) * np.abs(ex))
        C.record(kernel="mHC prompt (_hc_partial+_hc_finish)", case=f"real {layer}.hc_{which} rows={rows} collapse+{which}_norm", **m)


def t_router(layer, ne, topk, slots):
    w = tensor(f"{layer}.ffn.gate.weight")                   # fp16 [ne, 5120]
    bias = tensor(f"{layer}.ffn.gate.bias").astype(np.float32)  # fp16 in the pack, widened exactly
    wb, bb = Buf.of(w), Buf.of(bias)
    kc = 20
    for rows in (1, 6, 16):
        tile = (16, 1, 2) if rows <= 6 else (8, 1, 2)
        # the kit's own variant (recorded, or derived for every window size: variants.py)
        kp = k_of("_rowmm_parts", lambda c, k: k["N"] == ne and k["RB"] == rows and k["BN"] == tile[0])
        x = C.rand_bf16(rng, (rows, D), 1.0)
        s = Buf(rows * kc * ne * 4)
        a = dict(X=P(Buf.of(x), "*bf16"), xs=D, W=P(wb, "*fp16"), S=P(s, "*fp32"), rows=rows)
        grid = ((ne + tile[0] - 1) // tile[0], kc, 1)
        if isinstance(kp, dict) and "hash" in kp:
            C.run_variant(kp, grid, a)
        else:
            kk = dict(kp)
            kk.pop("rows", None)
            C.KIT.run("_rowmm_parts", grid, a, kk)
        kr = k_of("_route_parts", lambda c, k: k["NE"] == ne)
        pk, wt = Buf(rows * slots * 4), Buf(rows * slots * 4)
        C.KIT.run("_route_parts", (rows, 1, 1), dict(S=P(s, "*fp32"), BIAS=P(bb, "*fp32"), PICK=P(pk, "*i32"),
                  WTS=P(wt, "*fp32"), scale=1.5, shared_id=ne), kr)
        lg = C.from_bf16_bits(x) @ w.astype(np.float64).T
        # numerics.json router.noaux_tc: the band takes dot_prob_bound (lambda 6; products exact in fp32, m = 256 per
        # chunk + 20 chunk sums, the sequential worst case of the declared split-K path)
        dlg = 6 * np.sqrt(256 + kc) * TR.U32 * (np.abs(C.from_bf16_bits(x)) @ np.abs(w.astype(np.float64)).T) \
            + TR.U32 * np.abs(lg)
        TR.route_check("router decode (_rowmm_parts+_route_parts)", f"real {layer} rows={rows}",
                       pk.download((rows, slots), np.int32), wt.download((rows, slots), np.float32).astype(np.float64),
                       lg, bias, topk, slots, ne, 1.5, dlg)


def t_engram(layer):
    qk = (tensor(f"{layer}.engram.q_weight").astype(np.float64) * tensor(f"{layer}.engram.k_weight")).astype(np.float32)
    k = k_of("_engram_gate", lambda c, k: True)
    for rows in (1, 16, 256):
        h = streams(rows)
        kv = C.rand_bf16(rng, (rows, 5 * D), 0.7)
        out = Buf(rows * HC * D * 2)
        C.KIT.run("_engram_gate", (rows, HC, 1), dict(H=P(Buf.of(h), "*bf16"), KV=P(Buf.of(kv), "*bf16"),
                  QK=P(Buf.of(qk), "*fp32"), OUT=P(out, "*bf16"), eps=EPS), k)
        hs, kvs = C.from_bf16_bits(h), C.from_bf16_bits(kv)
        key, val = kvs[:, :HC * D].reshape(rows, HC, D), kvs[:, HC * D:]
        rstd = 1 / np.sqrt((hs * hs).mean(-1) + EPS) / np.sqrt((key * key).mean(-1) + EPS)
        prod = hs * qk.astype(np.float64) * key
        dot = prod.sum(-1) * rstd / np.sqrt(D)
        g = lambda d_: 1 / (1 + np.exp(-np.copysign(np.sqrt(np.maximum(np.abs(d_), 1e-6)), d_)))
        gate = g(dot)
        ddot = C.gamma(D + 2) * np.abs(prod).sum(-1) * rstd / np.sqrt(D) + C.gamma(16) * np.abs(dot)
        dg = np.maximum(np.abs(g(dot + ddot) - gate), np.abs(g(dot - ddot) - gate)) + C.gamma(16) * gate
        ex = hs + gate[..., None] * val[:, None, :]
        bound = np.abs(val[:, None, :]) * dg[..., None] + C.gamma(2) * (np.abs(hs) + np.abs(gate[..., None] * val[:, None, :]))
        m = C.cmp_rounded(C.from_bf16_bits(out.download((rows, HC, D), np.uint16)), ex, R.bf16(ex), C.ulp_bf16, bound=bound)
        C.record(kernel="_engram_gate", case=f"real {layer}.engram q*k rows={rows}", **m)


def t_markov():
    head = tensor("mtp.2.markov_head.head.weight")           # fp16 [V, 256]
    emb = tensor("mtp.2.markov_head.embed.weight")           # bf16 bits [V, 256]
    V, NC = head.shape[0], head.shape[0] // 2
    hb, eb = Buf.of(head), Buf.of(emb)
    kb = k_of("_bias", lambda c, k: True)
    none = np.full(V, -1, np.int32)
    for seg0 in (0, 1):
        toks = rng.integers(0, V, 4)
        out = Buf(4 * NC * 2)
        C.KIT.run("_bias", ((NC + 255) // 256, 1, 1), dict(TOK=P(Buf.of(toks.astype(np.int64)), "*i64"), t_stride=1,
                  SLOT=P(Buf.of(none), "*i32"), EMB=P(eb, "*bf16"), HEAD=P(hb, "*fp16"), OUT=P(out, "*fp16"), o_row=NC,
                  n_rows=4, n_cols=NC, seg0=seg0), kb)
        e = R.f16(C.from_bf16_bits(emb[toks]))
        hseg = head[seg0 * NC:(seg0 + 1) * NC].astype(np.float64)
        ref = e @ hseg.T
        m = C.cmp_rounded(out.download((4, NC), np.float16).astype(np.float64), ref, R.f16(ref), C.ulp_f16,
                          bound=C.gamma(256) * (np.abs(e) @ np.abs(hseg).T))
        m["fp16_overflow_in_e"] = int(np.isinf(e).sum())
        C.record(kernel="_bias", case=f"real mtp.2.markov_head seg0={seg0}", **m)


if __name__ == "__main__":
    t_norms("layers.2")
    t_norms("layers.20")
    t_norms("mtp.2")
    t_mhc("layers.2", "attn")
    t_mhc("layers.20", "ffn")
    t_mhc("mtp.2", "attn")
    t_router("layers.2", 384, 6, 7)
    t_router("layers.20", 384, 6, 7)
    t_router("mtp.2", 128, 3, 4)
    t_engram("layers.14")
    t_markov()
    H.sync()
    sys.exit(C.done())
