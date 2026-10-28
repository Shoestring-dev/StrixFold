/* w-tri: the indexer's scores of a prompt chunk for gfx1151 (wave32 WMMA), replacing the Triton _index_score_rows
 * (kernels.py at 222a9c0b) on the prompt path. ADAPTER.md section 6: group "aux", one by-value struct
 * (tf_ds_aux_index_args, abi.h), registered in registry.json.
 *
 * What it computes, for each row r and key t < n (32 index heads of dimension 128):
 *   score[r, t] = sum_h relu(q[r, h] . k[t]) * w[r, h] in fp32 (the head sum in a fixed order: the 8 heads
 *   16 g + 2 i + j of a lane in i order for g = 0 then g = 1, then the lane pair j = 0, 1), -inf at t >= vis[r];
 *   k[t] from the FP4 cache (byte j: element j | element j + 64 << 4; an E8M0 scale per 32 elements), decoded
 *   exactly to bf16; q, w bf16. flags bit 0: the int64 top-k keys (the score's bits in total order above, ~t below,
 *   -0 as +0) instead of fp32 scores; bit 1 (with keys): each 64-key tile's largest key into tmax [rows, nt].
 *   A key tile wholly at or past every row's visible count is written without its dots (all -inf).
 * Work: a workgroup a group of 8 rows and a range of the key tiles (grid y splits the tiles: few rows still fill the
 * GPU), each tile of 64 in turn (read and decoded once for the 8 rows, staged into one of two LDS buffers while the
 * other is in use: one barrier per tile); wave w: row w, all 32 heads, its Q fragments in registers (two 16-head
 * halves sharing each key fragment); Q K^T by WMMA.
 * A row's results depend on its own inputs only (row invariance), never on the row group or the split. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "abi.h"

typedef __bf16 tf_bf16x16 __attribute__((ext_vector_type(16)));
typedef float tf_f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t tf_u32x4 __attribute__((ext_vector_type(4)));
#define TF_GLOBAL __attribute__((address_space(1)))

namespace {
constexpr int IH = 32, ID = 128, RT = 8, NT = 64, KPAD = ID + 8;

template <class T> __device__ inline TF_GLOBAL T* gp(uint64_t a) { return (TF_GLOBAL T*)a; }

__device__ inline void lds_sync() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

__device__ inline float e2m1(uint32_t c) {
    const uint32_t m = c & 7u;
    const uint32_t bits = m >= 2u ? ((((m >> 1) + 126u) << 23) | ((m & 1u) << 22)) : (m == 1u ? 0x3F000000u : 0u);
    return __uint_as_float(bits | ((c & 8u) << 28));
}

// bf16 bits of a float whose low 16 bits are zero (an E2M1 value times a power of two), by truncation
__device__ inline uint32_t bf16_exact(float v) { return __float_as_uint(v) >> 16; }

typedef unsigned short tf_u16x2 __attribute__((ext_vector_type(2)));

// Four E2M1 codes (the low nibbles of c's bytes, element k in byte k) times 2^(e - 127) for 2 <= e <= 252 (o2: the
// exponent offset (e - 127) << 7 in both 16-bit halves), as two words of bf16 pairs (elements 0, 1 and 2, 3): the
// unscaled bf16 bytes by byte-permute tables, the sign or-ed in, the offset added to every nonzero value's exponent
// (no carry into the sign: the exponent stays within 1..254). Bits equal to bf16(e2m1 * 2^(e - 127)).
__device__ inline void fp4x4_e8(uint32_t c, uint32_t o2, uint32_t& lo, uint32_t& hi) {
    const uint32_t m = c & 0x07070707u;
    const uint32_t hb = __builtin_amdgcn_perm(0x40404040u, 0x3F3F3F00u, m) | ((c & 0x08080808u) << 4);
    const uint32_t lb = __builtin_amdgcn_perm(0xC0804000u, 0xC0800000u, m);
    const uint32_t nz = __builtin_amdgcn_perm(0xFFFFFFFFu, 0xFFFFFF00u, m);      // 0xFF where the magnitude is nonzero
    const uint32_t p0 = __builtin_amdgcn_perm(hb, lb, 0x05010400u), p1 = __builtin_amdgcn_perm(hb, lb, 0x07030602u);
    const uint32_t n0 = __builtin_amdgcn_perm(nz, nz, 0x01010000u) & o2;
    const uint32_t n1 = __builtin_amdgcn_perm(nz, nz, 0x03030202u) & o2;
    tf_u16x2 a, b, x, y;
    __builtin_memcpy(&a, &p0, 4);
    __builtin_memcpy(&b, &p1, 4);
    __builtin_memcpy(&x, &n0, 4);
    __builtin_memcpy(&y, &n1, 4);
    a += x;
    b += y;
    __builtin_memcpy(&lo, &a, 4);
    __builtin_memcpy(&hi, &b, 4);
}

template <class P> __device__ inline tf_bf16x16 frag(const P* p) {
    const tf_u32x4 lo = *reinterpret_cast<const tf_u32x4*>(p);
    const tf_u32x4 hi = *reinterpret_cast<const tf_u32x4*>(p + 8);
    uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    tf_bf16x16 v;
    __builtin_memcpy(&v, w, 32);
    return v;
}

__device__ inline tf_bf16x16 frag_g(const TF_GLOBAL __bf16* p) {
    const tf_u32x4 lo = *reinterpret_cast<const TF_GLOBAL tf_u32x4*>(p);
    const tf_u32x4 hi = *reinterpret_cast<const TF_GLOBAL tf_u32x4*>(p + 8);
    uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    tf_bf16x16 v;
    __builtin_memcpy(&v, w, 32);
    return v;
}

// topk_indices' int64 key of score s at column t
__device__ inline int64_t score_key(float s, int64_t t) {
    uint32_t bits = __float_as_uint(s);
    if (s == 0.f) bits = 0u;                                   // -0 as +0 (score + 0.0)
    const int32_t sb = (int32_t)bits;
    const uint32_t ord = (uint32_t)(sb < 0 ? (sb ^ 0x7FFFFFFF) : sb);
    return (int64_t)(((uint64_t)ord << 32) | (uint64_t)(0xFFFFFFFFu - (uint32_t)t));
}

// One staged key quarter of this thread: key t / 4, elements 32 (t % 4) .. + 32 (2 x 16 code bytes, one scale)
struct Quarter {
    tf_u32x4 c0, c1;
    uint32_t scale;
};
}  // namespace

extern "C" __global__ __launch_bounds__(256) void tf_ds_aux_index_rows_h32_d128(tf_ds_aux_index_args a) {
    __shared__ __attribute__((aligned(32))) __bf16 kc[2][NT][KPAD];   // two key tiles, bf16, rows padded (2 x 17 KB)

    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int r0 = blockIdx.x * RT, row = r0 + w;
    const bool live = row < a.rows;
    const bool keys = a.flags & 1, tmax_on = a.flags & 2;
    const int n = a.n, nt = a.nt;

    // this wave's Q fragments (row, head 16 g + l16, 8 k-steps) and head weights (head 16 g + 2 i + hi)
    tf_bf16x16 qf[2][8];
    float wv[2][8];
    int64_t vis = 0;
    if (live) {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const TF_GLOBAL __bf16* q = gp<const __bf16>(a.q) + ((int64_t)row * IH + g * 16 + l16) * ID;
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) qf[g][ks] = frag_g(q + ks * 16);
            const TF_GLOBAL uint16_t* wr = gp<const uint16_t>(a.w) + (int64_t)row * IH + g * 16;
#pragma unroll
            for (int i = 0; i < 8; ++i) wv[g][i] = __uint_as_float((uint32_t)wr[2 * i + hi] << 16);
        }
        vis = gp<const int64_t>(a.vis)[row];
    }
    // the group's largest visible count: tiles at or past it hold -inf for every row of the group
    int64_t vmax = 0;
#pragma unroll
    for (int i = 0; i < RT; ++i)
        if (r0 + i < a.rows) {
            const int64_t v = gp<const int64_t>(a.vis)[r0 + i];
            vmax = v > vmax ? v : vmax;
        }
    const int live_tiles = (int)((vmax + NT - 1) / NT) < nt ? (int)((vmax + NT - 1) / NT) : nt;

    const int sk = tid >> 2, sq = tid & 3;                    // the staged key and quarter of this thread
    auto load = [&](int kt, Quarter& qd) {
        const int64_t t = (int64_t)kt * NT + sk;
        if (t >= n) {
            qd.c0 = qd.c1 = tf_u32x4{0, 0, 0, 0};
            qd.scale = 0;
            return;
        }
        const TF_GLOBAL tf_u32x4* cp = reinterpret_cast<const TF_GLOBAL tf_u32x4*>(
            gp<const uint8_t>(a.k) + t * (ID / 2) + (sq & 1) * 32);
        qd.c0 = cp[0];
        qd.c1 = cp[1];
        qd.scale = gp<const uint8_t>(a.ks)[t * (ID / 32) + sq];
    };
    // decode a staged quarter exactly to bf16 into buffer b
    auto stage = [&](const Quarter& cur, int b) {
        const bool upper = sq >= 2;                           // elements 64..127: the high nibbles
        const uint32_t cw[8] = {cur.c0.x, cur.c0.y, cur.c0.z, cur.c0.w, cur.c1.x, cur.c1.y, cur.c1.z, cur.c1.w};
        uint32_t wd[16];
        const int e = (int)cur.scale;
        if (e >= 2 && e <= 252) {
            // the scale only moves the exponent: the E2M1 values' bf16 bits by table, plus (e - 127) on the exponent
            const uint32_t o16 = (uint32_t)((e - 127) << 7) & 0xFFFFu;
#pragma unroll
            for (int w = 0; w < 8; ++w)
                fp4x4_e8(upper ? cw[w] >> 4 : cw[w], o16 | (o16 << 16), wd[2 * w], wd[2 * w + 1]);
        } else {
            const float sc = __uint_as_float(cur.scale << 23);   // E8M0: 2^(byte - 127) (byte 0: 0, 255: inf)
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint32_t word = cw[i >> 1];
                const uint32_t b0 = (word >> (16 * (i & 1))) & 0xFFu, b1 = (word >> (16 * (i & 1) + 8)) & 0xFFu;
                wd[i] = bf16_exact(e2m1(upper ? b0 >> 4 : b0 & 15u) * sc) |
                        (bf16_exact(e2m1(upper ? b1 >> 4 : b1 & 15u) * sc) << 16);
            }
        }
        tf_u32x4* dst = reinterpret_cast<tf_u32x4*>(&kc[b][sk][sq * 32]);
#pragma unroll
        for (int v = 0; v < 4; ++v) dst[v] = tf_u32x4{wd[4 * v], wd[4 * v + 1], wd[4 * v + 2], wd[4 * v + 3]};
    };

    Quarter nxt;
    // this workgroup's key tiles tb .. te - 1: split blockIdx.y of gridDim.y (a chunk of few rows spreads its keys over
    // more workgroups; a key's results do not depend on the split); tiles tb .. le - 1 hold visible keys
    const int per = (nt + (int)gridDim.y - 1) / (int)gridDim.y, tb = (int)blockIdx.y * per;
    const int te = tb + per < nt ? tb + per : nt, le = live_tiles < te ? live_tiles : te;
    if (tb < le) {
        load(tb, nxt);
        stage(nxt, 0);
        if (tb + 1 < le) load(tb + 1, nxt);
    }
    lds_sync();

    for (int kt = tb; kt < te; ++kt) {
        const int64_t t0 = (int64_t)kt * NT;
        if (kt >= live_tiles) {
            // every row of the group sees none of this tile: -inf scores, their keys, the tile's maximum key
            if (live) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int64_t t = t0 + j * 16 + l16;
                    if (hi == 0 && t < n) {
                        if (keys) gp<int64_t>(a.out)[(int64_t)row * n + t] = score_key(-__builtin_inff(), t);
                        else gp<float>(a.out)[(int64_t)row * n + t] = -__builtin_inff();
                    }
                }
                if (keys && tmax_on && lane == 0)
                    gp<int64_t>(a.tmax)[(int64_t)row * nt + kt] = score_key(-__builtin_inff(), t0);
            }
            continue;
        }
        const int b = (kt - tb) & 1;
        // 1. S = Q K^T for the row's 32 heads and the tile's 64 keys (4 tiles of 16): relu * w, the head sum, the
        //    visible mask, keys and the tile's maximum
        if (live) {
            int64_t best = INT64_MIN;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                tf_f32x8 acc0 = tf_f32x8{0, 0, 0, 0, 0, 0, 0, 0}, acc1 = acc0;
#pragma unroll
                for (int ks = 0; ks < 8; ++ks) {
                    const tf_bf16x16 kf = frag(&kc[b][j * 16 + l16][ks * 16]);
                    acc0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf[0][ks], kf, acc0);
                    acc1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf[1][ks], kf, acc1);
                }
                float s = 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) s += fmaxf(acc0[i], 0.f) * wv[0][i];
#pragma unroll
                for (int i = 0; i < 8; ++i) s += fmaxf(acc1[i], 0.f) * wv[1][i];
                s += __shfl_xor(s, 16, 32);                       // the lane pair: heads 2 i and 2 i + 1
                const int64_t t = t0 + j * 16 + l16;
                if (t >= vis) s = -__builtin_inff();
                if (t < n) {
                    if (keys) {
                        const int64_t key = score_key(s, t);
                        if (hi == 0) gp<int64_t>(a.out)[(int64_t)row * n + t] = key;
                        best = key > best ? key : best;
                    } else if (hi == 0) {
                        gp<float>(a.out)[(int64_t)row * n + t] = s;
                    }
                }
            }
            if (keys && tmax_on) {
#pragma unroll
                for (int o = 1; o < 16; o <<= 1) {
                    const int64_t other = __shfl_xor(best, o, 32);
                    best = other > best ? other : best;
                }
                if (lane == 0) gp<int64_t>(a.tmax)[(int64_t)row * nt + kt] = best;
            }
        }
        // 2. stage the next live tile into the other buffer (last read before the previous barrier), then start
        //    loading the one after it
        if (kt + 1 < le) {
            stage(nxt, b ^ 1);
            if (kt + 2 < le) load(kt + 2, nxt);
        }
        lds_sync();
    }
}
