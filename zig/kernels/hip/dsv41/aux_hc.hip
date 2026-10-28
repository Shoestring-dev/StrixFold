/* w-tri: the mHC posted streams and mixes of a prompt chunk for gfx1151, replacing the Triton _hc_mix_pf (kernels.py
 * at 222a9c0b) on the prompt path. ADAPTER.md section 6: group "aux", one by-value struct (tf_ds_aux_hc_mix_args,
 * abi.h), registered in registry.json.
 *
 * What it computes, for each row r < rows (4 streams of D = 5120 columns; 40 mix blocks of 512 columns, block
 * b = 10 s + c holding columns 512 c .. + 512 of stream s):
 *   flags bit 0 (posted): y = bf16(sum_q g[q, r] in q order, fp32); the new streams xo[r, s] = bf16(fma(post[r, s], y,
 *   fma(comb[r, 3, s], x3, fma(comb[r, 2, s], x2, fma(comb[r, 0, s], x0, comb[r, 1, s] x1))))) (_hc_post's
 *   contraction), and the mixes below of xo; without it the mixes of x itself (xo, g, post, comb unused).
 *   part[r, b, m] = sum over the block's columns k in order of fma(xs[r, k], fn[m, 5120 s + k], .) for m < 24;
 *   part[r, b, 24] = the same of fma(xs, xs, .) (slots 25..31 untouched).
 * Work: a workgroup 32 rows x one 512-column chunk c of all four streams (each row's x, g read once), in steps of 8
 * rows x the whole chunk, so a row's 1 KB of a stream is read in one piece (128-column steps of 32 rows read 256-byte
 * pieces and ran at ~60% of a streaming kernel's bandwidth): every thread computes part of the step's streams into LDS
 * (and xo); then thread (s, row, group g) accumulates stream s's mixes 3 g .. 3 g + 2 (and the sum of squares) of the
 * step's row over the chunk's columns in order, in eight 64-column halves whose weights are staged in LDS as
 * wl[s][col][m] (loaded into registers two halves ahead). The next step's inputs load one item at each even half, after
 * the weights of two halves on, so a half's weights never wait on them (loads complete in order). Workgroups run
 * chunk-major (the ten chunks of a row group side by side, for DRAM locality); x, g and xo are streamed
 * non-temporally. A row's results depend on its own inputs only (row invariance), and each sum adds its terms in the
 * same column order as the 128-column version (the same bits).
 * Measured (p-pre2, gfx1151 warmed to ~2.1 GHz, world 2): 1,024 rows 0.66 ms (0.91 for the 128-column version; a
 * streaming kernel moving the same 126 MB takes 0.55 ms), 640 rows 0.43 (0.63), 128 rows 0.10 (0.21). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "abi.h"

typedef uint32_t tf_u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t tf_u32x2 __attribute__((ext_vector_type(2)));
typedef float tf_f32x4 __attribute__((ext_vector_type(4)));
#define TF_GLOBAL __attribute__((address_space(1)))

namespace {
constexpr int D = 5120, HC = 4, WIDE = HC * D, NC = 10, KB = 512, NB = HC * NC, RB = 32;
constexpr int RS = 8, NSTEP = RB / RS, PITCH = KB + 8; // HC_ROWSTEP: a step is 8 rows x the whole 512-column chunk
constexpr int NQ = KB / 4;                             // column quads of a row
constexpr int NI = RS * NQ / 256;                      // a thread's (row, quad) items of a step
constexpr int WP = 25;                                 // a column's 24 weights in wl, padded (bank spread of the stores)

template <class T> __device__ inline TF_GLOBAL T* gp(uint64_t a) { return (TF_GLOBAL T*)a; }

__device__ inline void lds_sync() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

// fp32 -> bf16 bits, round to nearest even; NaN stays NaN
__device__ inline uint32_t bf16_rne(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (u >> 16) | 0x40u;
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

__device__ inline float lo_f(uint32_t w) { return __uint_as_float(w << 16); }
__device__ inline float hi_f(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

// One (row, column quad) item of a step: the 4 streams' x and the ranks' partial sums of y
struct Item {
    tf_u32x2 x[HC];
    tf_f32x4 g0, g1;
};
}  // namespace

extern "C" __global__ __launch_bounds__(256) void tf_ds_aux_hc_mix_pf_d5120(tf_ds_aux_hc_mix_args a) {
    __shared__ __attribute__((aligned(16))) uint16_t xs[HC][RS][PITCH];   // a step's streams, bf16 (33 KB)
    __shared__ float pc[RB][20];                                         // comb [j][s] then post [s] of each row
    __shared__ float wl[HC][64][WP];                                     // a 64-column half's weights [s][col][m] (25 KB)

    const int tid = threadIdx.x;
    const int r0 = (blockIdx.x / NC) * RB, c = blockIdx.x % NC, rows = a.rows;
    const bool posted = a.flags & 1;
    const int nsteps = min(NSTEP, (rows - r0 + RS - 1) / RS);   // (uniform: the steps with rows)

    if (posted) {
        for (int i = tid; i < RB * 20; i += 256) {
            const int rl = i / 20, e = i % 20, row = r0 + rl;
            float v = 0.f;
            if (row < rows) v = e < 16 ? gp<const float>(a.comb)[(int64_t)row * 16 + e] : gp<const float>(a.post)[(int64_t)row * 4 + e - 16];
            pc[rl][e] = v;
        }
    }
    // a step's input item k: k * 256 + tid is row r0 + step RS + item / NQ, quad item % NQ (a wave: 128 columns of one
    // row); a row past the end reads the last row (never stored), so every wave issues the same loads (exact waits)
    auto load1 = [&](int step, int k, Item& it) {
        const int item = k * 256 + tid, rl = item / NQ, q = item % NQ, row = min(r0 + step * RS + rl, rows - 1);
        const int col = c * KB + 4 * q;
        const TF_GLOBAL uint16_t* xr = gp<const uint16_t>(a.x) + (int64_t)row * WIDE + col;
#pragma unroll
        for (int j = 0; j < HC; ++j) it.x[j] = __builtin_nontemporal_load(reinterpret_cast<const TF_GLOBAL tf_u32x2*>(xr + j * D));
        if (posted) {
            const TF_GLOBAL float* gr = gp<const float>(a.g) + (int64_t)row * D + col;
            it.g0 = __builtin_nontemporal_load(reinterpret_cast<const TF_GLOBAL tf_f32x4*>(gr));
            if (a.world > 1) it.g1 = __builtin_nontemporal_load(reinterpret_cast<const TF_GLOBAL tf_f32x4*>(gr + a.rs));
        }
    };
    auto load = [&](int step, Item (&it)[NI]) {
#pragma unroll
        for (int k = 0; k < NI; ++k) load1(step, k, it[k]);
    };
    // the step's streams into LDS (posted: computed, and stored to xo)
    auto compute = [&](int step, const Item (&it)[NI]) {
#pragma unroll
        for (int k = 0; k < NI; ++k) {
            const int item = k * 256 + tid, rl = item / NQ, q = item % NQ, rw = step * RS + rl, row = r0 + rw;
            if (row >= rows) continue;
            if (!posted) {
#pragma unroll
                for (int s = 0; s < HC; ++s) *reinterpret_cast<tf_u32x2*>(&xs[s][rl][4 * q]) = it[k].x[s];
                continue;
            }
            const int col = c * KB + 4 * q;
            float y[4] = {it[k].g0.x, it[k].g0.y, it[k].g0.z, it[k].g0.w};
            if (a.world > 1) {
                y[0] = y[0] + it[k].g1.x;
                y[1] = y[1] + it[k].g1.y;
                y[2] = y[2] + it[k].g1.z;
                y[3] = y[3] + it[k].g1.w;
            }
            float xv[HC][4];
#pragma unroll
            for (int j = 0; j < HC; ++j) {
                xv[j][0] = lo_f(it[k].x[j].x);
                xv[j][1] = hi_f(it[k].x[j].x);
                xv[j][2] = lo_f(it[k].x[j].y);
                xv[j][3] = hi_f(it[k].x[j].y);
            }
            TF_GLOBAL uint16_t* xo = gp<uint16_t>(a.xo) + (int64_t)row * WIDE + col;
#pragma unroll
            for (int s = 0; s < HC; ++s) {
                const float c0 = pc[rw][0 * 4 + s], c1 = pc[rw][1 * 4 + s], c2 = pc[rw][2 * 4 + s], c3 = pc[rw][3 * 4 + s];
                const float ps = pc[rw][16 + s];
                uint32_t ob[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float yb = __uint_as_float(bf16_rne(y[e]) << 16);
                    const float v = fmaf(ps, yb, fmaf(c3, xv[3][e], fmaf(c2, xv[2][e], fmaf(c0, xv[0][e], c1 * xv[1][e]))));
                    ob[e] = bf16_rne(v);
                }
                const tf_u32x2 packed = {ob[0] | (ob[1] << 16), ob[2] | (ob[3] << 16)};
                __builtin_nontemporal_store(packed, reinterpret_cast<TF_GLOBAL tf_u32x2*>(xo + s * D));
                *reinterpret_cast<tf_u32x2*>(&xs[s][rl][4 * q]) = packed;
            }
        }
    };

    // the mixes: thread (s, row rl, group mg) accumulates stream s's mixes 3 mg .. 3 mg + 2 (and its sum of squares) of
    // the step's row rl over the chunk's columns in order; a wave holds one stream's 4 rows x 8 groups
    const int s = tid >> 6, rl = (tid >> 3) & 7, mg = tid & 7;
    float acc[3], ss = 0.f;

    // the weights of one 64-column half (columns 64 hs .. + 64 of every stream), wl[s][col][m]: loaded into registers a
    // half ahead (6 x 16 bytes a thread), stored transposed when the previous half's mixes are done
    tf_f32x4 wr2[2][6];
    auto wload = [&](int hs, tf_f32x4 (&wr)[6]) {
#pragma unroll
        for (int k = 0; k < 6; ++k) {
            const int i = tid + 256 * k, m = i >> 6, s4 = (i >> 4) & 3, j4 = i & 15;
            wr[k] = *reinterpret_cast<const TF_GLOBAL tf_f32x4*>(gp<const float>(a.fn) + (int64_t)m * WIDE + s4 * D + c * KB +
                                                                hs * 64 + 4 * j4);
        }
    };
    auto wstore = [&](const tf_f32x4 (&wr)[6]) {
#pragma unroll
        for (int k = 0; k < 6; ++k) {
            const int i = tid + 256 * k, m = i >> 6, s4 = (i >> 4) & 3, j4 = i & 15;
            wl[s4][4 * j4 + 0][m] = wr[k].x;
            wl[s4][4 * j4 + 1][m] = wr[k].y;
            wl[s4][4 * j4 + 2][m] = wr[k].z;
            wl[s4][4 * j4 + 3][m] = wr[k].w;
        }
    };
    // the mixes over the chunk's columns 64 h .. + 64 (columns in order)
    auto mix_half = [&](int h) {
        const uint16_t* xr = &xs[s][rl][h * 64];
#pragma unroll 2
        for (int d8 = 0; d8 < 8; ++d8) {
            const tf_u32x4 v = *reinterpret_cast<const tf_u32x4*>(xr + d8 * 8);
            const float xv[8] = {lo_f(v.x), hi_f(v.x), lo_f(v.y), hi_f(v.y), lo_f(v.z), hi_f(v.z), lo_f(v.w), hi_f(v.w)};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float* w = &wl[s][d8 * 8 + j][3 * mg];
                acc[0] = fmaf(xv[j], w[0], acc[0]);
                acc[1] = fmaf(xv[j], w[1], acc[1]);
                acc[2] = fmaf(xv[j], w[2], acc[2]);
                ss = fmaf(xv[j], xv[j], ss);
            }
        }
    };

    // HC_PIPE: the next step's inputs one item at the even halves, the weights two halves ahead (two register sets),
    // so a half's weights never wait on the streaming loads issued after them
    Item cur[NI];
    load(0, cur);
    wload(0, wr2[0]);
    wload(1, wr2[1]);
    lds_sync();
    for (int step = 0; step < nsteps; ++step) {
        compute(step, cur);
        const bool more = step + 1 < nsteps;
        acc[0] = acc[1] = acc[2] = 0.f;
        ss = 0.f;
#pragma unroll
        for (int h = 0; h < KB / 64; ++h) {
            wstore(wr2[h & 1]);
            wload((h + 2) & 7, wr2[h & 1]);   // (past the last step: two halves' weights read for nothing)
            if ((h & 1) == 0 && more) load1(step + 1, h >> 1, cur[h >> 1]);
            lds_sync();
            mix_half(h);
            lds_sync();
        }
        const int row = r0 + step * RS + rl;
        if (row < rows) {
            TF_GLOBAL float* pr = gp<float>(a.part) + ((int64_t)row * NB + s * NC + c) * 32;
            pr[3 * mg + 0] = acc[0];
            pr[3 * mg + 1] = acc[1];
            pr[3 * mg + 2] = acc[2];
            if (mg == 0) pr[24] = ss;
        }
    }
}
