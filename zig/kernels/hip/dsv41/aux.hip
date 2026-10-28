/* w-tri's auxiliary dsv41 kernels for HIP (ADAPTER.md section 6: group "aux", abi "params", fixed-width parameters,
 * registered in registry.json). Ported from TensorFold's CUDA sources at strix-port-base (Apache-2.0):
 *   zig/kernels/cuda/dsv41_ops.cu (sha256 728ef4b0784143736945d8ecb44397f6c25af5db3445ab523ee99c0b8946b0a6):
 *     topk_i64, topk_indices, iota_vis, cand_fast, block_max, pool_pick, apply_pool and their score-key helper
 *   zig/kernels/cuda/dsv41_torch.cu (sha256 f212ce75443e414fcc580be56fcc75346d456963fdc246bfd6a1188096ad39bc):
 *     rms_norm, compress2, rope, bf16_scale, hc_mean4
 * Kernel bodies are the CUDA ones; changes: names (tf_ds_<op>_kernel -> tf_ds_aux_<op>), parameter types spelled with
 * fixed widths (long long -> int64_t, int -> int32_t: the same sizes, so ops.zig / exact.zig pack them unchanged),
 * __launch_bounds__ on every kernel (the block its Zig launcher uses, or the bound of its variable block), bf16 as HIP's
 * __hip_bfloat16 (its float conversion rounds to nearest even; __float2bfloat16_rn maps to it), and rms_norm's
 * full-warp __shfl_down_sync as the wave32 __shfl_down. The CUDA build used --fmad=false --ftz=false: the kit's flags
 * are -ffp-contract=off -fdenormal-fp-math=ieee, and every arithmetic operation here is an explicit _rn intrinsic or
 * plain IEEE operation as there. Validated against float64 references by tools/dsv41-hip/tri/t_native.py. */
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include "abi.h"

typedef __hip_bfloat16 tf_ds_bf16;
#define __float2bfloat16_rn(x) __float2bfloat16(x)

// topk_indices' key of score x at column j as an unsigned integer in the keys' signed order: x's bits (-0 as +0, as
// score + 0.0 makes it), ordered as signed integers, above the inverted column.
__device__ __forceinline__ uint64_t tf_ds_score_key(float x, int32_t j) {
    unsigned int bits = __float_as_uint(x);
    if (bits == 0x80000000u) bits = 0u;                        // + 0.0
    const int sb = (int)bits;
    const int ordered = sb < 0 ? (sb ^ 0x7FFFFFFF) : sb;
    const uint64_t key = ((uint64_t)(int64_t)ordered << 32) | (uint64_t)(0xFFFFFFFFu - (unsigned int)j);
    return key ^ 0x8000000000000000ull;                       // signed order as unsigned
}

// torch's keys.topk(k, sorted=False).values for the indexer's int64 keys, a row a block (1024 threads): the k largest
// values as one multiset in no particular order (_topk_finish sorts their indices). A radix select over the keys in
// signed order, eight 8-bit digits from the top: each digit's 256-bin histogram of the keys still matching the prefix,
// the bin holding the k-th largest found by a wave's suffix sums. It stops at the first digit whose bin holds exactly
// the keys still wanted (the keys are unique: score bits above, inverted index below, so this is the usual case after
// 2-4 digits); then every key above the prefix and every key on it is written. Equal keys (the pruned path's pad tile
// maxima) run to the last digit, and copies of the k-th fill the rest. Rows of at most 32768 keys are held in registers
// across the digits (32 a thread); longer rows are read again each digit. keys [rows, n] (row stride ks).
// (w-tri, 2026-10-19: the CUDA port's thread-0 bin scan and fixed eight digits took 7.5 ms for 2048 rows of 1K keys.)
namespace {
constexpr int TOPK_REG = 32;                                   // keys a thread holds (n <= 32 x 1024)

// the suffix bin of hist [256] holding the want-th largest: wave 0 only; returns (bin, keys above it)
__device__ inline void topk_bin(const unsigned int* hist, int want, int& bin, int& above) {
    const int lane = threadIdx.x & 31;
    unsigned int v[8], s = 0;                                  // lane l: bins 255 - 8 l - j, j = 0..7 (descending)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = hist[255 - 8 * lane - j];
        s += v[j];
    }
    unsigned int incl = s;                                     // inclusive prefix over lanes (lane 0 = the top bins)
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned int t = __shfl_up(incl, o, 32);
        if (lane >= o) incl += t;
    }
    const unsigned int excl = incl - s;
    const uint64_t hit = __ballot(incl >= (unsigned int)want && excl < (unsigned int)want);
    const int src = __ffsll((unsigned long long)hit) - 1;      // the lane whose bins reach want (exists: n >= k)
    unsigned int c = excl;
    int b = 0;
    if (lane == src) {
        b = 255 - 8 * lane - 7;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (c + v[j] >= (unsigned int)want) {
                b = 255 - 8 * lane - j;
                break;
            }
            c += v[j];
        }
    }
    bin = __shfl(b, src, 32);
    above = (int)__shfl(c, src, 32);
}
}  // namespace

extern "C" __global__ void __launch_bounds__(1024) tf_ds_aux_topk_i64(const int64_t* keys, int64_t ks, int32_t n, int32_t k,
                                                                           int64_t* top) {
    __shared__ unsigned int hist[256];
    __shared__ uint64_t prefix_s, mask_s;
    __shared__ int want_s, done_s, count_s;
    const int64_t* row = keys + (int64_t)blockIdx.x * ks;
    int64_t* out = top + (int64_t)blockIdx.x * k;
    const int tid = threadIdx.x;
    const bool regs = n <= TOPK_REG * 1024;
    uint64_t kr[TOPK_REG];
    if (regs) {
#pragma unroll
        for (int j = 0; j < TOPK_REG; ++j) {
            const int i = tid + j * 1024;
            kr[j] = i < n ? ((uint64_t)row[i] ^ 0x8000000000000000ull) : 0;
        }
    }
    if (tid == 0) {
        prefix_s = 0;
        mask_s = 0;
        want_s = k;
        done_s = 0;
        count_s = 0;
    }
    for (int d = 7; d >= 0; --d) {
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        if (done_s) break;
        const uint64_t prefix = prefix_s, mask = mask_s;
        if (regs) {
#pragma unroll
            for (int j = 0; j < TOPK_REG; ++j)
                if (tid + j * 1024 < n && (kr[j] & mask) == prefix) atomicAdd(&hist[(kr[j] >> (8 * d)) & 0xff], 1u);
        } else {
            for (int i = tid; i < n; i += 1024) {
                const uint64_t u = (uint64_t)row[i] ^ 0x8000000000000000ull;
                if ((u & mask) == prefix) atomicAdd(&hist[(u >> (8 * d)) & 0xff], 1u);
            }
        }
        __syncthreads();
        if (tid < 32) {
            const int want = want_s;
            int b, above;
            topk_bin(hist, want, b, above);
            if (tid == 0) {
                const int rest = want - above;                   // keys wanted from bin b
                want_s = rest;
                prefix_s = prefix | ((uint64_t)b << (8 * d));
                mask_s = mask | (0xffull << (8 * d));
                done_s = (int)hist[b] == rest;                   // bin b holds exactly the keys still wanted
            }
        }
        __syncthreads();
    }
    // every key above the prefix (fewer than k), then the keys on it (exactly the rest, or copies of the k-th)
    const uint64_t prefix = prefix_s, mask = mask_s;
    if (regs) {
#pragma unroll
        for (int j = 0; j < TOPK_REG; ++j)
            if (tid + j * 1024 < n && (kr[j] & mask) > prefix)
                out[atomicAdd(&count_s, 1)] = (int64_t)(kr[j] ^ 0x8000000000000000ull);
    } else {
        for (int i = tid; i < n; i += 1024) {
            const uint64_t u = (uint64_t)row[i] ^ 0x8000000000000000ull;
            if ((u & mask) > prefix) out[atomicAdd(&count_s, 1)] = row[i];
        }
    }
    __syncthreads();
    if (regs) {
#pragma unroll
        for (int j = 0; j < TOPK_REG; ++j)
            if (tid + j * 1024 < n && (kr[j] & mask) == prefix) {
                const int at = atomicAdd(&count_s, 1);
                if (at < k) out[at] = (int64_t)(kr[j] ^ 0x8000000000000000ull);
            }
    } else {
        for (int i = tid; i < n; i += 1024) {
            const uint64_t u = (uint64_t)row[i] ^ 0x8000000000000000ull;
            if ((u & mask) == prefix) {
                const int at = atomicAdd(&count_s, 1);
                if (at < k) out[at] = row[i];
            }
        }
    }
}

// kernels.topk_indices(score, k), then torch.where(top < vis, top, -1) (model.py attention_k, the candidate pool's
// layers), a row a block (1024 threads): each score's int64 key (its bits, -0 as +0 (score + 0.0), ordered as signed
// integers, above the inverted column: a total order, ties to the lower column), the k largest keys by a radix select
// (eight 8-bit digits from the top), their columns ascending (a bitonic sort), a column at or past vis[r] as -1.
// score [rows, n] fp32 (row stride ss), vis [rows] int64, out [rows, k] int64; k <= 2048 and k <= n (the candidate
// pool's top blocks: k = 2048).
extern "C" __global__ void __launch_bounds__(1024) tf_ds_aux_topk_indices(const float* score, int64_t ss, int32_t n,
                                                                              int32_t k, const int64_t* vis,
                                                                              int64_t* out) {
    __shared__ unsigned int hist[256];
    __shared__ uint64_t prefix_s, mask_s;
    __shared__ int want_s, count_s;
    __shared__ int64_t cols[2048];
    const float* row = score + (int64_t)blockIdx.x * ss;
    if (threadIdx.x == 0) {
        prefix_s = 0;
        mask_s = 0;
        want_s = k;
        count_s = 0;
    }
    __syncthreads();
    for (int d = 7; d >= 0; --d) {
        for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        const uint64_t prefix = prefix_s, mask = mask_s;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint64_t u = tf_ds_score_key(row[i], i);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> (8 * d)) & 0xff], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int want = want_s;
            int b = 255;
            for (; b > 0; --b) {
                if ((int)hist[b] >= want) break;
                want -= (int)hist[b];
            }
            want_s = want;
            prefix_s = prefix | ((uint64_t)b << (8 * d));
            mask_s = mask | (0xffull << (8 * d));
        }
        __syncthreads();
    }
    const uint64_t kth = prefix_s;   // the keys are unique: exactly k at or above it
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (tf_ds_score_key(row[i], i) >= kth) {
            const int at = atomicAdd(&count_s, 1);
            if (at < k) cols[at] = i;
        }
    }
    int p = 1;
    while (p < k) p <<= 1;
    for (int i = k + threadIdx.x; i < p; i += blockDim.x) cols[i] = 0x7FFFFFFFFFFFFFFFll;
    for (int size = 2; size <= p; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int i = threadIdx.x; i < p; i += blockDim.x) {   // (p up to 2048: two elements a thread)
                const int j = i ^ stride;
                if (j > i) {
                    const bool up = (i & size) == 0;
                    const int64_t a = cols[i], b = cols[j];
                    if ((a > b) == up) {
                        cols[i] = b;
                        cols[j] = a;
                    }
                }
            }
        }
    }
    __syncthreads();
    const int64_t v = vis[blockIdx.x];
    int64_t* o = out + (int64_t)blockIdx.x * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) o[i] = cols[i] < v ? cols[i] : -1;
}

// The indexer's selection when its top-k takes every key it scans (nb <= index_topk): kernels.topk_indices of all nb
// keys is 0 .. nb - 1 whatever the scores, then the visible mask: out[r, j] = j if j < vis[r] else -1. vis [rows] int64,
// out [rows, nb] int64; grid (blocks, rows).
extern "C" __global__ void __launch_bounds__(256) tf_ds_aux_iota_vis(const int64_t* vis, int64_t* out, int32_t nb) {
    const int r = blockIdx.y;
    const int64_t v = vis[r];
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < nb; j += gridDim.x * blockDim.x) {
        out[(int64_t)r * nb + j] = j < v ? (int64_t)j : -1;
    }
}

// rounds.py _candidates_fast where the pool takes every block (contexts up to nblocks * bsize keys): a row's block mask
// (torch's bool, as u8) = the block's maximum (amax: a NaN wins) above -inf, or the block is the row's newest,
// (vis - 1) // bsize (floor division). score [rows, nb * bsize] fp32 (row stride ss), vis [rows], out [rows, nb] (row
// stride os); grid (blocks, rows).
extern "C" __global__ void __launch_bounds__(256) tf_ds_aux_cand_fast(const float* score, int64_t ss, int32_t nb, int32_t bsize,
                                                  const int64_t* vis, uint8_t* out, int64_t os) {
    const int r = blockIdx.y;
    const int64_t t = vis[r] - 1;
    const int64_t last = t >= 0 ? t / bsize : -((-t + bsize - 1) / bsize);
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nb; b += gridDim.x * blockDim.x) {
        const float* s = score + (int64_t)r * ss + (int64_t)b * bsize;
        bool nan = false, above = false;
        for (int j = 0; j < bsize; ++j) {
            const float x = s[j];
            if (x != x) nan = true;
            else if (x > -__int_as_float(0x7f800000)) above = true;
        }
        out[(int64_t)r * os + b] = (uint8_t)((!nan && above) || b == last);
    }
}

// model.py _candidates / rounds.py _candidate_blocks, their first step: a row's block maxima of the scores (amax over
// blocks of bsize, the last block padded with -inf, a NaN wins), its newest block, (vis - 1) // bsize (floor
// division), pinned to +inf. score [rows, width] fp32 (row stride ss), vis [rows] int64, out [rows, nb] fp32 (row
// stride os), nb = ceil(width / bsize); grid (blocks, rows).
extern "C" __global__ void __launch_bounds__(256) tf_ds_aux_block_max(const float* score, int64_t ss, int32_t width, int32_t bsize,
                                                  const int64_t* vis, float* out, int64_t os, int32_t nb) {
    const int r = blockIdx.y;
    const int64_t t = vis[r] - 1;
    const int64_t last = t >= 0 ? t / bsize : -((-t + bsize - 1) / bsize);
    const float ninf = -__int_as_float(0x7f800000);
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nb; b += gridDim.x * blockDim.x) {
        const float* s = score + (int64_t)r * ss + (int64_t)b * bsize;
        float m = ninf;
        bool nan = false;
        for (int j = 0; j < bsize && b * bsize + j < width; ++j) {
            const float x = s[j];
            if (x != x) nan = true;
            else if (x > m) m = x;
        }
        float v = nan ? __int_as_float(0x7fc00000) : m;
        if (b == last) v = __int_as_float(0x7f800000);
        out[(int64_t)r * os + b] = v;
    }
}

// The pool from a row's top blocks idx [rows, k] (kernels.topk_indices of the block maxima bmax [rows, nb], row stride
// bs): model.py _candidates' mask (u8 [rows, nb], row stride ms: zeros, then each picked block whose maximum is above
// -inf) and/or rounds.py _candidate_blocks' list (int32 [rows, k]: the block, or -1 when its maximum is not above -inf).
// A row a block of threads.
extern "C" __global__ void __launch_bounds__(1024) tf_ds_aux_pool_pick(const float* bmax, int64_t bs, int32_t nb, const int64_t* idx, int32_t k,
                                                  uint8_t* mask, int64_t ms, int32_t* cblk) {
    const int r = blockIdx.x;
    const float ninf = -__int_as_float(0x7f800000);
    if (mask != nullptr) {
        for (int b = threadIdx.x; b < nb; b += blockDim.x) mask[(int64_t)r * ms + b] = 0;
        __syncthreads();
    }
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        const int64_t c = idx[(int64_t)r * k + j];
        const bool ok = bmax[(int64_t)r * bs + c] > ninf;
        if (mask != nullptr && ok) mask[(int64_t)r * ms + c] = 1;
        if (cblk != nullptr) cblk[(int64_t)r * k + j] = ok ? (int)c : -1;
    }
}

// model.py apply_candidates: -inf at every position whose block the row's pool leaves out (mask u8 [rows, >= ceil(width
// / bsize)], row stride ms), in place. score [rows, width] fp32 (row stride ss); grid (blocks, rows).
extern "C" __global__ void __launch_bounds__(256) tf_ds_aux_apply_pool(float* score, int64_t ss, int32_t width, int32_t bsize, const uint8_t* mask,
                                                   int64_t ms) {
    const int r = blockIdx.y;
    const float ninf = -__int_as_float(0x7f800000);
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < width; j += gridDim.x * blockDim.x) {
        if (!mask[(int64_t)r * ms + j / bsize]) score[(int64_t)r * ss + j] = ninf;
    }
}

// ops.rms_norm(x, w, eps) of rows of D bf16 values (D a multiple of 4, at least 128): out = bf16(w * (x * r)),
// r = rsqrt(mean(x * x) + eps). The mean is torch's reduce_kernel's for a contiguous last dim (Reduce.cuh,
// vectorize_input): lane l of the row's L lanes (blockDim.x) sums the 4-wide vectors l, l + L, ... into four
// accumulators, combined in order; lanes above a warp fold in through shared memory (offsets L/2 .. 32), then a warp
// shuffle with decreasing offsets; times `factor` (torch's float(rows) / (rows * D)). Block (L, H): a row a y.
extern "C" __global__ void __launch_bounds__(512) tf_ds_aux_rms_norm(const tf_ds_bf16* x, int64_t ldx, const tf_ds_bf16* w,
                                                 tf_ds_bf16* out, int64_t ldo, int32_t rows, int32_t D, float factor,
                                                 float eps) {
    extern __shared__ float shared[];   // L * H floats when L > 32
    __shared__ float rs[64];            // each row's r
    const int L = blockDim.x;
    const int lane = threadIdx.x;
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const bool live = row < rows;
    const tf_ds_bf16* xr = x + (int64_t)(live ? row : 0) * ldx;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    if (live) {
        for (int idx = lane; idx * 4 + 3 < D; idx += L) {
            const float a = __bfloat162float(xr[idx * 4 + 0]);
            const float b = __bfloat162float(xr[idx * 4 + 1]);
            const float c = __bfloat162float(xr[idx * 4 + 2]);
            const float d = __bfloat162float(xr[idx * 4 + 3]);
            acc0 = __fadd_rn(acc0, __fmul_rn(a, a));
            acc1 = __fadd_rn(acc1, __fmul_rn(b, b));
            acc2 = __fadd_rn(acc2, __fmul_rn(c, c));
            acc3 = __fadd_rn(acc3, __fmul_rn(d, d));
        }
    }
    float v = __fadd_rn(__fadd_rn(__fadd_rn(acc0, acc1), acc2), acc3);
    int dim_x = L;
    if (L > 32) {
        const int base = threadIdx.x + threadIdx.y * L;
        shared[base] = v;
        for (int offset = L / 2; offset >= 32; offset >>= 1) {
            __syncthreads();
            if (lane < offset && lane + offset < L) {
                v = __fadd_rn(v, shared[base + offset]);
                shared[base] = v;
            }
        }
        dim_x = 32;
    }
    __syncthreads();
    for (int offset = dim_x >> 1; offset > 0; offset >>= 1) v = __fadd_rn(v, __shfl_down(v, offset, 32));
    if (lane == 0) rs[threadIdx.y] = rsqrtf(__fadd_rn(__fmul_rn(v, factor), eps));
    __syncthreads();
    if (!live) return;
    const float r = rs[threadIdx.y];
    tf_ds_bf16* o = out + (int64_t)row * ldo;
    for (int j = lane; j < D; j += L) {
        const float y = __fmul_rn(__bfloat162float(xr[j]), r);
        o[j] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(w[j]), y));
    }
}

// _compress at ratio 2 after its two fp32 projections: kv, score [G, 2, C] -> out bf16 [G, C] =
// bf16((kv * score.softmax(dim=1)).sum(1)) at torch's rounding points: the spatial softmax over the pair (max, the
// sum of exp(s - max) from 0, then exp(s - max) / sum), the products, and the pair's sum as reduce_kernel's
// thread_reduce_impl adds it (four accumulators from 0, two of them used, combined in order).
extern "C" __global__ void __launch_bounds__(256) tf_ds_aux_compress2(const float* kv, const float* score, tf_ds_bf16* out, int32_t G, int32_t C) {
    const int64_t n = (int64_t)G * C;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t g = i / C, c = i % C;
        const float s0 = score[(g * 2) * C + c];
        const float s1 = score[(g * 2 + 1) * C + c];
        float m = -3.402823466e+38f;   // numeric_limits<float>::lowest()
        m = m < s0 ? s0 : m;
        m = m < s1 ? s1 : m;
        float sum = 0.f;
        sum = __fadd_rn(sum, expf(__fsub_rn(s0, m)));
        sum = __fadd_rn(sum, expf(__fsub_rn(s1, m)));
        const float p0 = __fdiv_rn(expf(__fsub_rn(s0, m)), sum);
        const float p1 = __fdiv_rn(expf(__fsub_rn(s1, m)), sum);
        const float y0 = __fmul_rn(kv[(g * 2) * C + c], p0);
        const float y1 = __fmul_rn(kv[(g * 2 + 1) * C + c], p1);
        float t = __fadd_rn(__fadd_rn(0.f, y0), __fadd_rn(0.f, y1));
        t = __fadd_rn(t, 0.f);
        t = __fadd_rn(t, 0.f);
        out[i] = __float2bfloat16_rn(t);
    }
}

// ops.rope_(x[..., off : off + 2 * half], f) of n rows (bf16, row stride ldx), in place: (a + b i)(c + d i), d negated
// for the inverse (f.conj()), each part rounded to bf16. f = complex(cos, sin) at table row pos[r] (fp32 [*, half]).
// torch's complex<float> multiply is a*c - b*d, a*d + b*c, and its CUDA build contracts each into one fused
// multiply-add over the other product: fma(a, c, -(b*d)) and fma(b, c, a*d) (the bytes of torch.mul on GB10, the
// four other roundings of the pair differ: zrec_torchlab.py).
extern "C" __global__ void __launch_bounds__(256) tf_ds_aux_rope(tf_ds_bf16* x, int64_t ldx, int32_t off, const float* cos,
                                             const float* sin, const int64_t* pos, int32_t n, int32_t half, int32_t inverse) {
    const int64_t total = (int64_t)n * half;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = i / half, k = i % half;
        tf_ds_bf16* p = x + r * ldx + off + 2 * k;
        const float a = __bfloat162float(p[0]);
        const float b = __bfloat162float(p[1]);
        const int64_t t = pos[r] * half + k;
        const float c = cos[t];
        const float d = inverse ? -sin[t] : sin[t];
        p[0] = __float2bfloat16_rn(__fmaf_rn(a, c, -__fmul_rn(b, d)));
        p[1] = __float2bfloat16_rn(__fmaf_rn(b, c, __fmul_rn(a, d)));
    }
}

// The indexer's head weights (model.py attention_k): wts = wl.to(bf16) * s, wl fp32: rounded to bf16, then times the
// Python scalar as torch's mul with a scalar computes it (fp32 opmath: float(bf16) * float(s)), rounded to bf16 again.
extern "C" __global__ void __launch_bounds__(256) tf_ds_aux_bf16_scale(const float* in, tf_ds_bf16* out, uint64_t count, float s) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += uint64_t(gridDim.x) * blockDim.x)
        out[i] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(__float2bfloat16_rn(in[i])), s));
}

// A DSpark tap as the prompt forward takes it, h.to(fp32).mean(1).to(bf16) over the 4 streams: torch's reduce_kernel
// over the middle dim (a thread an output: four accumulators from 0, one value each, combined in order; too few
// values a thread to split them over the block), times MeanOps' factor 0.25, rounded to bf16. h [rows, 4, D] bf16,
// out [rows, D] (row stride os).
extern "C" __global__ void __launch_bounds__(256) tf_ds_aux_hc_mean4(const tf_ds_bf16* h, tf_ds_bf16* out, int64_t os, int32_t rows,
                                                 int32_t D) {
    const int64_t total = (int64_t)rows * D;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = i / D, d = i % D;
        const tf_ds_bf16* x = h + r * 4 * D + d;
        const float a0 = __fadd_rn(0.f, __bfloat162float(x[0]));
        const float a1 = __fadd_rn(0.f, __bfloat162float(x[D]));
        const float a2 = __fadd_rn(0.f, __bfloat162float(x[2 * D]));
        const float a3 = __fadd_rn(0.f, __bfloat162float(x[3 * D]));
        const float s = __fadd_rn(__fadd_rn(__fadd_rn(a0, a1), a2), a3);
        out[r * os + d] = __float2bfloat16_rn(__fmul_rn(s, 0.25f));
    }
}
