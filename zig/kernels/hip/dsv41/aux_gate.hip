/* w-tri: the MoE router gate's logits of a prompt chunk for gfx1151 (wave32 WMMA), replacing prompt.zig moe()'s
 * toF32(x) + f16ToF32(gate_w) + hipBLAS sgemm (torch: x.float() @ gate_w.float().t()). ADAPTER.md section 6: group
 * "aux", one by-value struct (tf_ds_aux_gate_args, abi.h), registered in registry.json.
 *
 * What it computes: out[r, e] = sum over k < 5120 of x[r, k] * w[e, k] in fp32, x bf16, w fp16. Each fp16 weight is
 * split exactly into two bf16 (hi: its top 8 significant bits by truncation, lo: the remaining 3 bits), so every
 * product of the bf16 WMMA is exact; the WMMA's fp32 accumulation (not IEEE round-to-nearest: see t_wmma.py) adds
 * 640 times a row, within gamma(5120) of the exact sum of |x w| (the contract torch's fp32 GEMM has too).
 * Work: a workgroup BM rows x BN outputs, k in steps of 32 through two LDS buffers (one barrier a step; the next
 * step's x and w loaded into registers during this one); waves of (BM / WM) x (BN / WN) accumulator tiles. Two
 * builds: the router gates (ne a multiple of 64: 128 x 64, 8 waves) and the indexer's head weights (weights_proj,
 * fp16 [32, 5120], ne = 32: 32 x 32, 4 waves). A row's results depend on its own x only (row invariance). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "abi.h"

typedef __bf16 tf_bf16x16 __attribute__((ext_vector_type(16)));
typedef float tf_f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t tf_u32x4 __attribute__((ext_vector_type(4)));
typedef _Float16 tf_h8 __attribute__((ext_vector_type(8)));
#define TF_GLOBAL __attribute__((address_space(1)))

namespace {
constexpr int D = 5120, BK = 32, PITCH = BK + 8;    // rows of 80 bytes: conflict-free fragments

template <class T> __device__ inline TF_GLOBAL T* gp(uint64_t a) { return (TF_GLOBAL T*)a; }

__device__ inline void lds_sync() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

// 16 consecutive bf16 of LDS (16-byte aligned) as a fragment
__device__ inline tf_bf16x16 frag(const __bf16* p) {
    const tf_u32x4 lo = *reinterpret_cast<const tf_u32x4*>(p);
    const tf_u32x4 hi = *reinterpret_cast<const tf_u32x4*>(p + 8);
    uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    tf_bf16x16 v;
    __builtin_memcpy(&v, w, 32);
    return v;
}

// eight fp16 as eight bf16 high parts (truncated) and the exact remainders, packed in pairs
__device__ inline void split8(tf_u32x4 raw, tf_u32x4& hi, tf_u32x4& lo) {
    tf_h8 h;
    __builtin_memcpy(&h, &raw, 16);
    uint32_t hb[8], lb[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float f = (float)h[i];                            // exact
        const uint32_t t = __float_as_uint(f) & 0xFFFF0000u;    // its top 8 significant bits
        hb[i] = t >> 16;
        lb[i] = __float_as_uint(f - __uint_as_float(t)) >> 16;  // at most 3 significant bits: exact in bf16
    }
    hi = tf_u32x4{hb[0] | (hb[1] << 16), hb[2] | (hb[3] << 16), hb[4] | (hb[5] << 16), hb[6] | (hb[7] << 16)};
    lo = tf_u32x4{lb[0] | (lb[1] << 16), lb[2] | (lb[3] << 16), lb[4] | (lb[5] << 16), lb[6] | (lb[7] << 16)};
}
}  // namespace

// The gate GEMM: a workgroup BM rows x BN experts, WM x WN waves of (BM / WM) x (BN / WN) outputs each.
template <int BM, int BN, int WM, int WN>
__device__ inline void gate_logits(const tf_ds_aux_gate_args& a) {
    constexpr int NT = 32 * WM * WN, TM = BM / WM / 16, TN = BN / WN / 16;   // threads; a wave's 16 x 16 tiles
    constexpr int XC = BM * (BK / 8), WC = BN * (BK / 8);                  // 16-byte pieces of a step's x and w
    static_assert(XC % NT == 0 && WC % NT == 0, "staging pieces divide the threads");
    __shared__ __attribute__((aligned(16))) __bf16 xs[2][BM][PITCH];
    __shared__ __attribute__((aligned(16))) __bf16 wh[2][BN][PITCH];
    __shared__ __attribute__((aligned(16))) __bf16 wl[2][BN][PITCH];

    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int r0 = blockIdx.x * BM, e0 = blockIdx.y * BN, rows = a.rows, ne = a.ne;
    const int wm = w % WM, wn = w / WM;

    // staging: piece p of x: row p / 4, k 8 (p % 4) .. + 8; piece p of w: expert p / 4, k 8 (p % 4) .. + 8
    tf_u32x4 rx[XC / NT], rw[WC / NT];
    auto load = [&](int st) {
#pragma unroll
        for (int i = 0; i < XC / NT; ++i) {
            const int p = tid + i * NT, r = p >> 2, k = (p & 3) * 8;
            const TF_GLOBAL uint16_t* src = gp<const uint16_t>(a.x) + (int64_t)(r0 + r) * D + st * BK + k;
            rx[i] = r0 + r < rows ? *reinterpret_cast<const TF_GLOBAL tf_u32x4*>(src) : tf_u32x4{0, 0, 0, 0};
        }
#pragma unroll
        for (int i = 0; i < WC / NT; ++i) {
            const int p = tid + i * NT, e = p >> 2, k = (p & 3) * 8;
            const TF_GLOBAL uint16_t* src = gp<const uint16_t>(a.w) + (int64_t)(e0 + e) * D + st * BK + k;
            rw[i] = *reinterpret_cast<const TF_GLOBAL tf_u32x4*>(src);
        }
    };
    auto store = [&](int b) {
#pragma unroll
        for (int i = 0; i < XC / NT; ++i) {
            const int p = tid + i * NT;
            *reinterpret_cast<tf_u32x4*>(&xs[b][p >> 2][(p & 3) * 8]) = rx[i];
        }
#pragma unroll
        for (int i = 0; i < WC / NT; ++i) {
            const int p = tid + i * NT;
            tf_u32x4 h, l;
            split8(rw[i], h, l);
            *reinterpret_cast<tf_u32x4*>(&wh[b][p >> 2][(p & 3) * 8]) = h;
            *reinterpret_cast<tf_u32x4*>(&wl[b][p >> 2][(p & 3) * 8]) = l;
        }
    };

    tf_f32x8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = tf_f32x8{0, 0, 0, 0, 0, 0, 0, 0};

    constexpr int NST = D / BK;
    load(0);
    store(0);
    load(1);
    lds_sync();
    for (int st = 0; st < NST; ++st) {
        const int b = st & 1;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            tf_bf16x16 af[TM], bh[TN], bl[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) af[i] = frag(&xs[b][16 * (TM * wm + i) + l16][16 * ks]);
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                bh[j] = frag(&wh[b][16 * (TN * wn + j) + l16][16 * ks]);
                bl[j] = frag(&wl[b][16 * (TN * wn + j) + l16][16 * ks]);
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[i], bh[j], acc[i][j]);
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[i], bl[j], acc[i][j]);
                }
        }
        if (st + 1 < NST) {
            store(b ^ 1);
            if (st + 2 < NST) load(st + 2);
        }
        lds_sync();
    }
    // D layout: lane l holds expert 16 j + l % 16, rows 2 v + l / 16 of tile (i, j)
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int r = r0 + 16 * (TM * wm + i) + 2 * v + hi;
            if (r >= rows) continue;
#pragma unroll
            for (int j = 0; j < TN; ++j)
                gp<float>(a.out)[(int64_t)r * ne + e0 + 16 * (TN * wn + j) + l16] = acc[i][j][v];
        }
}

// the router gates (ne a multiple of 64): 128 rows x 64 experts a workgroup, 4 x 2 waves of 32 x 32 (p-pre2: 96 / 48
// workgroups at 1,024 rows instead of 48 / 16 x 3: 0.31-0.33 ms vs 0.42, the same bits; every output's k order is the
// tiling's own, gatebench ~/tfp/ppre/bench/out/gate1)
extern "C" __global__ __launch_bounds__(256) void tf_ds_aux_gate_logits_d5120(tf_ds_aux_gate_args a) {
    gate_logits<128, 64, 4, 2>(a);
}

// the indexer's head weights (ne = 32): 32 rows x 32 heads a workgroup, 2 x 2 waves of 16 x 16
extern "C" __global__ __launch_bounds__(128) void tf_ds_aux_gate_logits_n32_d5120(tf_ds_aux_gate_args a) {
    gate_logits<32, 32, 2, 2>(a);
}
