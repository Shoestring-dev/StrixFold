/* w-tri: the attention heads' RoPE for gfx1151, replacing the Triton _rope_heads (kernels.py at 222a9c0b) bit for bit.
 * ADAPTER.md section 6: group "aux", one by-value struct (tf_ds_aux_rope_heads_args, abi.h), registered in
 * registry.json.
 *
 * What it computes, in place, for each row r and head h of x bf16 [rows, h, hd]: the last rd dims as pairs
 * (x_e, x_o) = (x[hd - rd + 2 j], x[hd - rd + 2 j + 1]), j < rd / 2, rotated by position pos[r] with the fp32 tables
 * cos / sin [positions, rd / 2] (flags bit 0: inverse, sin negated): re = fma(x_e, c, -(x_o s)) and im = fma(x_e, s,
 * x_o c), or im = fma(x_o, c, x_e s) when inverse, each rounded to bf16 (nearest even): _rope_heads' compiled
 * contractions (found from its bits: t_hiprope compares every recorded variant bit for bit), so the bits equal the
 * Triton kernel's.
 * Work: a thread four pairs (16 bytes of x, 16 of cos and of sin), consecutive threads along a row's heads; no
 * thread depends on another (row invariance). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "abi.h"

typedef uint32_t tf_u32x4 __attribute__((ext_vector_type(4)));
typedef float tf_f32x4 __attribute__((ext_vector_type(4)));
#define TF_GLOBAL __attribute__((address_space(1)))

namespace {
template <class T> __device__ inline TF_GLOBAL T* gp(uint64_t a) { return (TF_GLOBAL T*)a; }

// fp32 -> bf16 bits, round to nearest even; NaN stays NaN
__device__ inline uint32_t bf16_rne(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (u >> 16) | 0x40u;
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}
}  // namespace

extern "C" __global__ __launch_bounds__(256) void tf_ds_aux_rope_heads(tf_ds_aux_rope_heads_args a) {
    const int quads = a.rd / 8;                                // four pairs a thread
    const int64_t t = (int64_t)blockIdx.x * 256 + threadIdx.x;
    const int64_t per_row = (int64_t)a.h * quads;
    if (t >= (int64_t)a.rows * per_row) return;
    const int64_t r = t / per_row;
    const int rem = (int)(t - r * per_row), head = rem / quads, q = rem - head * quads;
    const int64_t p = gp<const int64_t>(a.pos)[r];
    const tf_f32x4 c = *reinterpret_cast<const TF_GLOBAL tf_f32x4*>(gp<const float>(a.cos) + p * (a.rd / 2) + 4 * q);
    tf_f32x4 s = *reinterpret_cast<const TF_GLOBAL tf_f32x4*>(gp<const float>(a.sin) + p * (a.rd / 2) + 4 * q);
    const bool inv = a.flags & 1;
    if (inv) s = -s;
    TF_GLOBAL tf_u32x4* xp = reinterpret_cast<TF_GLOBAL tf_u32x4*>(
        gp<uint16_t>(a.x) + (r * a.h + head) * a.hd + (a.hd - a.rd) + 8 * q);
    const tf_u32x4 v = *xp;
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    const float cs[4] = {c.x, c.y, c.z, c.w}, sn[4] = {s.x, s.y, s.z, s.w};
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float xe = __uint_as_float(w[i] << 16), xo = __uint_as_float(w[i] & 0xFFFF0000u);
        const float re = fmaf(xe, cs[i], -(xo * sn[i]));
        const float im = inv ? fmaf(xo, cs[i], xe * sn[i]) : fmaf(xe, sn[i], xo * cs[i]);
        o[i] = bf16_rne(re) | (bf16_rne(im) << 16);
    }
    *xp = tf_u32x4{o[0], o[1], o[2], o[3]};
}
