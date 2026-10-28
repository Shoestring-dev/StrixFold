"""Independent float64 references for the DeepSeek-V4.1 auxiliary operators, written from the mathematical definitions
(DeepSeek's inference/model.py and kernel.py at deepseek-ai/DeepSeek-V4.1-Flash-1017 1bae0e10: RMSNorm, apply_rotary_emb,
hc_mixes / hc_split_sinkhorn / hc_pre / hc_post, Gate, Engram, act_quant / fp4_act_quant, sparse_attn, Indexer,
select_candidate_blocks, the DSpark Markov head) -- not from the Triton sources being tested. Arithmetic is float64;
format conversions the model defines (a tensor stored as bf16 / fp16 / fp8 / fp4) are applied at those points with
ml_dtypes' round-to-nearest-even casts (an implementation independent of both the CUDA and the AMD kernels).
"""

import ml_dtypes
import numpy as np

BF16 = ml_dtypes.bfloat16
F8 = ml_dtypes.float8_e4m3fn
F4 = ml_dtypes.float4_e2m1fn


def _round_sig(x, sig_bits, min_exp, max_val):
    """fp64 -> nearest value with `sig_bits` significant bits (round half to even), subnormal step 2^(min_exp -
    sig_bits + 1), overflow to inf: a single rounding straight from fp64 (ml_dtypes and numpy go through fp32 and can
    double-round)."""
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)                                        # |m| in [0.5, 1), x = m 2^e
    e = np.maximum(e, min_exp + 1)                            # subnormal range: fixed step
    q = np.ldexp(np.rint(np.ldexp(x, sig_bits - e)), e - sig_bits)
    return np.where(np.isfinite(x) & (np.abs(q) > max_val), np.copysign(np.inf, x), q)


def bf16(x):
    """fp64 -> bf16 (round to nearest even), as fp64."""
    return _round_sig(x, 8, -126, 3.3895313892515355e38)


def f16(x):
    """fp64 -> fp16 (round to nearest even), as fp64."""
    return _round_sig(x, 11, -14, 65504.0)


def f32(x):
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


def e4m3(x):
    """fp64 -> E4M3 (fn) nearest-even -> fp64 (callers clamp to +-448)."""
    return np.asarray(x, dtype=np.float64).astype(F8).astype(np.float64)


def e2m1(x):
    """fp64 -> E2M1 nearest-even -> fp64 (callers clamp to +-6)."""
    return np.asarray(x, dtype=np.float64).astype(F4).astype(np.float64)


def pow2_ceil(v):
    """2^ceil(log2 v) for v > 0 (exact for powers of two)."""
    v = np.asarray(v, dtype=np.float64)
    m, e = np.frexp(v)                 # v = m 2^e, m in [0.5, 1)
    return np.ldexp(1.0, np.where(m == 0.5, e - 1, e))


# ---- norms and rotations ------------------------------------------------------------------------------------------
def rmsnorm(x, w, eps):
    """DeepSeek RMSNorm: bf16(w * x / sqrt(mean(x^2) + eps)), x [..., D]."""
    x = np.asarray(x, np.float64)
    r = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return bf16(np.asarray(w, np.float64) * (x * r))


def hadamard128():
    h = np.array([[1.0]])
    while h.shape[0] < 128:
        h = np.block([[h, h], [h, -h]])
    return h


H128 = hadamard128()


def exl3_rot(y, suh):
    """EXL3's input rotation of rows y [R, D] (D a multiple of 128): each 128 block (y * suh) @ H / sqrt(128), fp16."""
    y = np.asarray(y, np.float64)
    R, D = y.shape
    v = (y * np.asarray(suh, np.float64)).reshape(R, D // 128, 128)
    return f16((v @ H128.T) / np.sqrt(128.0)).reshape(R, D)


def rope_pairs(x, cos, sin, inverse=False):
    """apply_rotary_emb on adjacent pairs of x [..., 2n] with cos/sin [..., n] (fp64 arithmetic, no rounding)."""
    x = np.asarray(x, np.float64)
    xe, xo = x[..., 0::2], x[..., 1::2]
    s = -sin if inverse else sin
    out = np.empty_like(x)
    out[..., 0::2] = xe * cos - xo * s
    out[..., 1::2] = xe * s + xo * cos
    return out


def fp8_qd_blocks(v, block):
    """act_quant(..., round_scale, inplace) on the last dim: blocks of `block`, s = 2^ceil(log2(max(amax,1e-4)/448)),
    e4m3(clamp(v / s, +-448)) * s."""
    v = np.asarray(v, np.float64)
    sh = v.shape
    b = v.reshape(*sh[:-1], sh[-1] // block, block)
    s = pow2_ceil(np.maximum(np.abs(b).max(-1, keepdims=True), 1e-4) / 448.0)
    return (e4m3(np.clip(b / s, -448, 448)) * s).reshape(sh)


def kv_norm_rope(y, w, cos_tab, sin_tab, pos, eps, rd, quant):
    """The window KV row: RMSNorm (bf16), RoPE on the last rd dims (bf16), FP8 quant-dequant in 32-blocks."""
    n = rmsnorm(y, w, eps)
    out = n.copy()
    c = cos_tab[pos][:, None, :] if n.ndim == 3 else cos_tab[pos]
    s = sin_tab[pos][:, None, :] if n.ndim == 3 else sin_tab[pos]
    out[..., -rd:] = bf16(rope_pairs(n[..., -rd:], np.asarray(c, np.float64), np.asarray(s, np.float64)))
    if quant:
        out = bf16(fp8_qd_blocks(out, 32))
    return out


def rope_heads(x, cos_tab, sin_tab, pos, rd, inverse):
    """x [R, H, HD]: RoPE on the last rd dims of every head (bf16)."""
    out = np.asarray(x, np.float64).copy()
    c = np.asarray(cos_tab[pos], np.float64)[:, None, :]
    s = np.asarray(sin_tab[pos], np.float64)[:, None, :]
    out[..., -rd:] = bf16(rope_pairs(out[..., -rd:], c, s, inverse))
    return out


# ---- mHC (hyper-connections) ------------------------------------------------------------------------------------
def hc_mixes_raw(x, fn):
    """x [R, hc, D] -> (dots [R, 24] = x_flat @ fn^T, sumsq [R]) in fp64."""
    xf = np.asarray(x, np.float64).reshape(x.shape[0], -1)
    return xf @ np.asarray(fn, np.float64).T, np.sum(xf * xf, axis=1)


def hc_split_sinkhorn(mix, scale, base, hc, iters, eps):
    """mix [R, (2+hc)*hc] (already scaled by rsqrt(mean(x^2)+eps)) -> pre [R,hc], post [R,hc], comb [R,hc,hc]."""
    mix = np.asarray(mix, np.float64)
    scale = np.asarray(scale, np.float64)
    base = np.asarray(base, np.float64)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    pre = sig(mix[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * sig(mix[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    cl = (mix[:, 2 * hc:] * scale[2] + base[2 * hc:]).reshape(-1, hc, hc)
    c = np.exp(cl - cl.max(-1, keepdims=True))
    c = c / c.sum(-1, keepdims=True) + eps
    c = c / (c.sum(-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(-1, keepdims=True) + eps)
        c = c / (c.sum(-2, keepdims=True) + eps)
    return pre, post, c


def hc_collapse(x, pre):
    """hc_pre's collapse: bf16(sum_j pre[j] x_j), x [R, hc, D], pre [R, hc]."""
    return bf16(np.einsum("rj,rjd->rd", np.asarray(pre, np.float64), np.asarray(x, np.float64)))


def hc_post(y, x, post, comb):
    """hc_post: out[k] = bf16(post[k] y + sum_j comb[j, k] x_j); y [R, D], x [R, hc, D]."""
    y = np.asarray(y, np.float64)
    return bf16(np.asarray(post, np.float64)[:, :, None] * y[:, None, :]
                + np.einsum("rjk,rjd->rkd", np.asarray(comb, np.float64), np.asarray(x, np.float64)))


# ---- routing ----------------------------------------------------------------------------------------------------
def gate(logits, bias, topk, route_scale):
    """Gate (sqrtsoftplus, noaux_tc): scores = sqrt(softplus(logits)); picks = topk(scores + bias) (ties: lower id);
    weights = scores[picks] / (sum + 1e-20) * route_scale. Returns picks [R, topk], weights, scores, the selection
    margins (kth - (k+1)th of scores + bias)."""
    lg = np.asarray(logits, np.float64)
    sc = np.sqrt(np.log1p(np.exp(-np.abs(lg))) + np.maximum(lg, 0))
    sel = sc + np.asarray(bias, np.float64)
    order = np.lexsort((np.arange(sel.shape[1])[None, :].repeat(sel.shape[0], 0), -sel), axis=1)
    picks = order[:, :topk]
    w = np.take_along_axis(sc, picks, 1)
    srt = np.take_along_axis(sel, order, 1)
    margin = srt[:, topk - 1] - srt[:, topk]
    return picks, w / (w.sum(1, keepdims=True) + 1e-20) * route_scale, sc, margin


# ---- FP4 caches -------------------------------------------------------------------------------------------------
def fp4_quant_rows(x, block, e4m3_scale):
    """fp4_act_quant of rows x [N, D] (fp32 compute as kernel.py's fp4_quant_kernel: amax / 6 and x / s are fp32
    divisions, then the casts round to nearest even): scale per block = E4M3(fp32(max(amax, 6*2^-9) / 6)), or
    2^ceil(log2(fp32(max(amax, 6*2^-126) / 6))); codes E2M1 of clamp(fp32(x / s), +-6). Returns (dequantized values,
    E2M1 nibbles [N, D], scale bytes [N, D/block])."""
    x = np.asarray(x, np.float64)
    N, D = x.shape
    b = x.reshape(N, D // block, block)
    amax = np.abs(b).max(-1)
    if e4m3_scale:
        s = e4m3(f32(np.maximum(amax, 6 * 2.0 ** -9) / 6.0))
        sbytes = s.astype(F8).view(np.uint8)
    else:
        s = pow2_ceil(f32(np.maximum(amax, 6 * 2.0 ** -126) / 6.0))
        sbytes = (np.log2(s).astype(np.int64) + 127).astype(np.uint8)
    q = e2m1(np.clip(f32(b / s[..., None]), -6, 6))
    codes = q.astype(F4).view(np.uint8).reshape(N, D) & 0xF
    codes = np.where((codes & 7) == 0, 0, codes).astype(np.uint8)   # TensorFold's cache encoding: zero is 0x0, never -0
    return (q * s[..., None]).reshape(N, D), codes, sbytes


def fp4_pack(codes):
    """E2M1 nibbles [N, D] -> bytes [N, D/2]: byte j = element j (low nibble) | element j + D/2 (high)."""
    half = codes.shape[1] // 2
    return (codes[:, :half] | (codes[:, half:] << 4)).astype(np.uint8)


def fp4_unpack(codes_packed, scales, block, e4m3_scale):
    """bytes [N, D/2] + scale bytes [N, D/block] -> fp64 [N, D]."""
    lo = codes_packed & 0xF
    hi = codes_packed >> 4
    nib = np.concatenate([lo, hi], axis=1)
    vals = np.ascontiguousarray(nib, dtype=np.uint8).view(F4).astype(np.float64)
    if e4m3_scale:
        s = scales.view(F8).astype(np.float64)
    else:
        s = np.ldexp(1.0, scales.astype(np.int64) - 127)
    N, D = vals.shape
    return (vals.reshape(N, D // block, block) * s[..., None]).reshape(N, D)


# ---- top-k ------------------------------------------------------------------------------------------------------
def topk_sorted(score, k):
    """Indices of the k largest scores (ties: lower index), ascending; plus the kth-(k+1)th margin."""
    s = np.asarray(score, np.float64)
    order = np.lexsort((np.arange(s.shape[-1])[None, :].repeat(s.shape[0], 0), -s), axis=1)
    top = np.sort(order[:, :k], axis=1)
    srt = np.take_along_axis(s, order, 1)
    margin = srt[:, k - 1] - srt[:, k] if s.shape[1] > k else np.full(s.shape[0], np.inf)
    return top, margin
