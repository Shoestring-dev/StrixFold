"""The prompt GEMM _gemm (exl3/prefill.py's matmul after rot_in and unpack: the prompt path's EXL3 linear past 128 rows)
-- every kit variant, recorded and derived (variants.py: each (K, N, OUT dtype, o_stride) the prompt path sends
through it) -- against float64 under numerics.json exl3.linear's acceptance, on this kernel's stage of the path:
y = ((xh @ W_q) @ H) * SCALE * svh, H the 128 x 128 Sylvester blocks, xh the rotated fp16 rows and W_q the decoded
fp16 weight taken exactly (rot_in's fp16 rounding happens before this kernel).

Declared path: WMMA fp16 x fp16 with fp32 accumulation over K (acc fp32_wmma, u = 2^-23: numerics.json wmma_rule),
the accumulator split into bf16 top + rest (2^-16 |acc|), their products with H in a 256-term WMMA chain, then fp32
* SCALE * svh (3 roundings with SCALE's own) and the output rounding (bf16 2^-8; fp32 none). With A = |xh| @ |W_q|
and a = xh @ W_q exact, per output j of 128-block b:
  B_j = SCALE |svh_j| sum_{i in b} [g(K) A_i + (2^-16 + g(256)) (1 + 2^-7) (|a_i| + g(K) A_i)] + (gamma_3 + u_out) |y_j|
  (g(m) = gamma_m at u = 2^-23); P_j the same with lambda sqrt(m) u for g(m), lambda = 6.
Accept: B everywhere, P exceeded on at most 1e-6 of the outputs (none under 1e6 outputs), no nonfinite value, nrmse
within exl3.linear's statistical limit (1e-3, + 2.5e-3 for a bf16 output), and no write outside the N columns of a
row whose stride o_stride is wider. Each launch runs on its own variant (exact selection, as the strict mode)."""

import json
import sys

import numpy as np

import common as C
import hiprt as H
import refs as R
from hiprt import Buf, P

LAUNCHES = C.launches()
NREC = len(json.loads((C.ROOT / "upstream" / "launches.json").read_text())["_gemm"])
rng = np.random.default_rng(4096)
C.log_open("t_gemm")
UW = 2.0 ** -23                       # fp32_wmma unit roundoff
LAM = 6.0
SCALE = 1.0 / np.sqrt(128.0)
HB = Buf.of(C.bf16_bits(R.H128))


def gw(m):
    return C.gamma(m, UW)


def run(case, xh, w, svh, m, out_dt):
    k = {n: (v if not isinstance(v, dict) else float(v["float"])) for n, v in case["consts"].items()}
    K, N = k["K"], k["N"]
    os_ = case["scalars"]["o_stride"]
    esz = 4 if out_dt == "float32" else 2
    out = Buf(m * os_ * esz)
    out.fill(0x7F)                                            # a sentinel outside the N columns
    svb = Buf.of(svh)
    args = dict(X=P(Buf.of(xh), "*fp16"), W=P(Buf.of(w), "*fp16"), H=P(HB, "*bf16"), SVH=P(svb, "*fp16"),
                BIAS=P(svb, "*fp16"), OUT=P(out, "*fp32" if out_dt == "float32" else "*bf16"), M=m, o_stride=os_)
    var = C.KIT.find("_gemm", args, k, exact_only=True)
    C.KIT.run("_gemm", ((m + 127) // 128 * (N // 128), 1, 1), args, k, variant=var)
    if out_dt == "float32":
        raw = out.download((m, os_), np.uint32)
        full, sentinel = raw.view(np.float32).astype(np.float64), 0x7F7F7F7F
    else:
        raw = out.download((m, os_), np.uint16)
        full, sentinel = C.from_bf16_bits(raw), 0x7F7F
    untouched = bool(np.all(raw[:, N:] == sentinel)) if os_ > N else True
    return full[:, :N], untouched


def t_gemm():
    for ci, c in enumerate(LAUNCHES["_gemm"]):
        K, N = c["consts"]["K"], c["consts"]["N"]
        out_dt = c["tensors"]["OUT"][0]
        os_ = c["scalars"]["o_stride"]
        tag0 = f"{'rec' if ci < NREC else 'derived'}{ci} K={K} N={N} OUT={out_dt} o_stride={os_}"
        w = rng.standard_normal((K, N)).astype(np.float16)
        svh = (rng.standard_normal(N) * 0.02).astype(np.float16)
        w64 = w.astype(np.float64)
        rows_set = (129, 1100) if K * N <= 5120 * 5120 else (129,)
        for m in rows_set:
            x = rng.standard_normal((m, K))
            x[:, rng.choice(K, 4, replace=False)] *= 40.0          # outlier channels (exl3.linear's expected cases)
            xh = x.astype(np.float16)
            try:
                got, untouched = run(c, xh, w, svh, m, out_dt)
            except KeyError as e:
                C.record(kernel="_gemm", case=f"{tag0} M={m}: exact variant in the kit", ok=False, error=str(e)[:300])
                continue
            x64 = xh.astype(np.float64)
            a = x64 @ w64
            A = np.abs(x64) @ np.abs(w64)
            nb = N // 128
            y0 = (a.reshape(m, nb, 128) @ R.H128).reshape(m, N)
            sv = svh.astype(np.float64)
            y = y0 * SCALE * sv
            sa = (np.abs(a) + gw(K) * A).reshape(m, nb, 128).sum(-1)       # per block
            sA = A.reshape(m, nb, 128).sum(-1)
            u_out = 2.0 ** -8 if out_dt == "bfloat16" else 0.0
            tail = (C.gamma(3) + u_out) * np.abs(y)
            split = (2.0 ** -16) * (1 + 2.0 ** -7) * sa
            hard = SCALE * np.abs(sv) * np.repeat(gw(K) * sA + split + gw(256) * (1 + 2.0 ** -7) * sa, 128, 1) + tail
            prob = SCALE * np.abs(sv) * np.repeat(LAM * np.sqrt(K) * UW * sA + split
                                                  + LAM * np.sqrt(256) * UW * (1 + 2.0 ** -7) * sa, 128, 1) + tail
            err = np.abs(got - y)
            nonfinite = int((~np.isfinite(got)).sum())
            exceed = int((err > prob).sum())
            allowed = 0 if got.size < 1e6 else int(1e-6 * got.size)
            nrmse = float(np.linalg.norm(got - y) / np.linalg.norm(y))
            nlim = 1e-3 + (2.5e-3 if out_dt == "bfloat16" else 0.0)
            ok = (nonfinite == 0 and float((err / hard).max()) <= 1.0 and exceed <= allowed and nrmse <= nlim
                  and untouched)
            C.record(kernel="_gemm", case=f"{tag0} M={m}", ok=bool(ok), bound_ratio=float((err / hard).max()),
                     prob_ratio=float((err / prob).max()), prob_exceed=exceed, nrmse=nrmse, nrmse_limit=nlim,
                     nonfinite=nonfinite, columns_past_n_untouched=untouched, n=int(got.size))
            if m == 1100:
                # row invariance (information only: numerics.json wmma_rule allows no exact gate through WMMA)
                g129, _ = run(c, xh[:129], w, svh, 129, out_dt)
                C.record(kernel="_gemm", case=f"{tag0} rows 0-128 at M=1100 vs M=129 (bits)", ok=None,
                         rows_differing=int(np.any(g129 != got[:129], 1).sum()))


if __name__ == "__main__":
    t_gemm()
    H.sync()
    sys.exit(C.done())
