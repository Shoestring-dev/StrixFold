/* w-exl3's EXL3 kernels for the dsv41 family on HIP (ADAPTER.md section 6: group "exl3", registered in registry.json;
 * exl3_linear.zig, exl3_prefill.zig and exl3_experts.zig launch them, hip_load.zig binds them). Each id keeps the
 * parameter list, launch geometry and documented semantics of the CUDA kernel of TensorFold's served build
 * (src/tensorfold/cuda/exl3 at 222a9c0b: linear_grouped.cu, linear.cu, experts.cu, experts_grouped.cuh): the same
 * inputs read, the same outputs and scratch written, every counter left zero, the epoch / ready protocol of the decode
 * experts, and the epilogues' arithmetic in their order (input rotation, Hadamard, svh, bias, RoPE, SwiGLU, combines).
 *
 * What differs is the matrix product. A 16 x 16 weight tile is decoded once per wave, exactly (MUL1: state *
 * 0x83DCD12D, byte sum, fp16(h * 0x1EEE + 0xC931) in one fused fp16 operation; bit-identical to w-exl3's CPU oracle and
 * to w-ref's codebook), and applied to every row of a decode / verify window (glinear below 64 rows, grouped_cp) with
 * v_dot2_f32_f16, each row in the same order whatever the window's width (row invariance), or to 16 rows at a time
 * with v_wmma_f32_16x16x16_f16 in the prompt paths (fp16 operands, fp32 accumulation either way). The fp32 sums run
 * in another order than mma.m16n8k16's, so the outputs meet the numerics contract (w-ref numerics.json exl3.linear,
 * exl3.experts_grouped) rather than the CUDA build's bits. Split-K partials and wave sums are added in a fixed order:
 * every output is deterministic and independent of what else shares the launch.
 *
 * gfx11.5 WMMA fragments (wave32): A lane l, half r -> A[row l & 15][k r]; B lane l, half r -> B[k r][col l & 15];
 * C lane l, float r -> C[row 2 r + (l >> 4)][col l & 15]; A and B are replicated across the two half-waves.
 * Tile format (format.py, after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): K2 half-bits a value, 4 K2 u32 words a
 * tile, an MSB-first bit stream whose value p is the 16-bit window ending at E(p) (mod 128 K2), placed at row
 * 2 (l % 4) + (j & 1) + 8 ((j >> 1) & 1), column l / 4 + 8 (j >> 2) for l = p / 8, j = p % 8.
 *
 * No PDL on AMD (griddepcontrol and the L2 discard / prefetch hints have no counterpart; launches on a stream run in
 * order). Compiled with the kit's flags (-ffp-contract=off): every fused multiply-add below is written as fmaf. */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include "abi.h"

namespace {

typedef __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16 f16x16;
typedef __attribute__((__vector_size__(8 * sizeof(float)))) float f32x8;
typedef _Float16 h2 __attribute__((ext_vector_type(2)));

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)
enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

// ---- conversions, loads and stores ---------------------------------------------------------------------------------

// clang 23 / HIP 7.15 compiles fp16(a * b) of fp32 a, b as v_fma_mixlo_f16 / v_fma_mixhi_f16: ONE rounding of the exact
// product to fp16 where the source (and the CUDA build) rounds to fp32 first, then to fp16; the two differ on fp16 ties
// (measured: rot_out 1 ulp off on 1 of ~8k values). An fp32 value made opaque first keeps both roundings.
__device__ __forceinline__ float f32_value(float x) {
    asm volatile("" : "+v"(x));
    return x;
}
__device__ __forceinline__ __half f2h(float f) { return __float2half_rn(f32_value(f)); }
__device__ __forceinline__ float bf16_bits_to_f(uint32_t b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ uint32_t f_to_bf16_bits(float f) { return __bfloat16_as_ushort(__float2bfloat16(f32_value(f))); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16(f32_value(x))); }
__device__ __forceinline__ uint32_t f_to_h_bits(float f) { return __half_as_ushort(f2h(f)); }

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(p) + i);
        if (dtype == BF16) {
            v[0] = bf16_bits_to_f(u.x & 0xffffu); v[1] = bf16_bits_to_f(u.x >> 16);
            v[2] = bf16_bits_to_f(u.y & 0xffffu); v[3] = bf16_bits_to_f(u.y >> 16);
        } else {
            v[0] = __half2float(__ushort_as_half(u.x & 0xffffu)); v[1] = __half2float(__ushort_as_half(u.x >> 16));
            v[2] = __half2float(__ushort_as_half(u.y & 0xffffu)); v[3] = __half2float(__ushort_as_half(u.y >> 16));
        }
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        *reinterpret_cast<uint2*>(static_cast<uint16_t*>(p) + i) =
            make_uint2(f_to_bf16_bits(v[0]) | f_to_bf16_bits(v[1]) << 16, f_to_bf16_bits(v[2]) | f_to_bf16_bits(v[3]) << 16);
    } else {
        *reinterpret_cast<uint2*>(static_cast<uint16_t*>(p) + i) =
            make_uint2(f_to_h_bits(v[0]) | f_to_h_bits(v[1]) << 16, f_to_h_bits(v[2]) | f_to_h_bits(v[3]) << 16);
    }
}

// A device address as a global (address space 1) pointer. Pointers made from the ABI's u64 fields or from the experts'
// pointer tables are generic, and generic loads compile to flat_load, which counts in lgkmcnt as well as vmcnt: every
// s_waitcnt lgkmcnt(0) before an LDS read would then also wait for the next step's weight prefetch (found by w-pf).
template <typename T>
__device__ __forceinline__ const __attribute__((address_space(1))) T* gl(const T* p) {
    return (const __attribute__((address_space(1))) T*)p;
}

// a coherent (L2) load and the acquire / release pair of the decode experts' readiness flags
__device__ __forceinline__ float ld_cg(const float* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ int ld_acquire(const int* p) { return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ void st_release(int* p, int v) { __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT); }

// ---- 128-point Walsh-Hadamard, the input rotation and the output epilogue ------------------------------------------

// Unscaled Walsh-Hadamard transform of 128 fp32 values, 4 a lane (elements 4 lane + j), in decode.cuh's fixed butterfly
// order (multiply by 1 / sqrt(128) after).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor(v[j], m, 32);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// One row's 128 rotated inputs ((x * suh) @ H) / sqrt(128) in fp32 (linear_common.cuh rot128_in: the first butterfly
// takes the products as fma(x0, s0, +-(x1 s1)) and fma(x2, s2, +-(x3 s3))).
__device__ __forceinline__ void rot128_in(float (&v)[4], const float (&s)[4], int lane) {
    const float p1 = v[1] * s[1], p3 = v[3] * s[3];
    const float a = fmaf(v[0], s[0], p1), b = fmaf(v[0], s[0], -p1);
    const float c = fmaf(v[2], s[2], p3), d = fmaf(v[2], s[2], -p3);
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor(v[j], m, 32);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE;
}

// The finished outputs of one row's 128 columns from their fp32 sums (linear_common.cuh finish): H / sqrt(128), * svh,
// + bias (the last product and the bias as one fma).
__device__ __forceinline__ void finish(float (&v)[4], int lane, const __half* svh, const __half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float t = v[j] * HAD_SCALE, sv = __half2float(svh[col + j]);
        v[j] = bias ? fmaf(t, sv, __half2float(bias[col + j])) : t * sv;
    }
}

// ---- the tile decode -------------------------------------------------------------------------------------------------

// E(p): one past the last stream bit of value p's 16-bit window (format.py stream_ends; decode.cuh stream_end).
template <int K2>
__host__ __device__ constexpr int stream_end(int p) {
    return (K2 & 1) ? ((p + 1) * K2 - ((p + 1) & 1)) / 2 : (p + 1) * (K2 / 2);
}

// A lane (column c = lane & 15, k half h = lane >> 4) owns the value pairs p0 = 32 (c % 8) + 8 q + 4 (c / 8) + 2 h,
// p0 + 1 (q = 0..3): rows 8 h + 2 q and 8 h + 2 q + 1 of column c. The pair's two windows lie in the 64 bits of words
// hi, lo (lo = the word after hi, wrapping), starting o bits into hi; the second window starts D bits after the first.
// hi, lo, sh: the 32 stream bits from the first window's start are alignbit(w[hi], w[lo], sh) (lo = hi and sh = 0
// when the window starts a word: v_alignbit_b32 takes its shift mod 32).
template <int K2>
struct LaneWin {
    static constexpr int TW = 4 * K2;
    static constexpr int D = stream_end<K2>(1) - stream_end<K2>(0);
    int hi[4], lo[4], sh[4];
    __device__ __forceinline__ explicit LaneWin(int lane) {
        const int c = lane & 15, h = lane >> 4;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int p0 = 32 * (c & 7) + 8 * q + 4 * (c >> 3) + 2 * h;
            const int s = (stream_end<K2>(p0) - 16 + 128 * K2) % (128 * K2);
            const int o = s & 31;
            hi[q] = s >> 5;
            lo[q] = o ? (hi[q] + 1) % TW : hi[q];
            sh[q] = (32 - o) & 31;
        }
    }
};

// x * 0x83DCD12D mod 2^32 of a 16-bit state. EXL3_MUL24=1: two 24-bit multiplies and a shift-add instead of
// v_mul_lo_u32 (measured: +1-5% at 1 row, -2-10% at 6 rows from its extra registers; off by default).
__device__ __forceinline__ uint32_t mul1_hash(uint32_t w) {
#if defined(EXL3_MUL24) && EXL3_MUL24
    // the barrier keeps the compiler from folding the shifted product back into a v_mul_lo_u32
    uint32_t b = __umul24(w, 0x83DCu);
    asm("" : "+v"(b));
    return __umul24(w, 0xD12Du) + (b << 16);
#else
    return w * 0x83DCD12Du;
#endif
}

// Two consecutive values from the 32 stream bits u at the first window's start -> packed fp16 pair (low half the
// first): MUL1's exact decode, fp16(0x6400 + bytesum(w * 0x83DCD12D)) * fp16(0x1EEE) + fp16(0xC931) in one fused fp16
// multiply-add.
template <int D>
__device__ __forceinline__ uint32_t decode_pair(uint32_t u) {
    const uint32_t w0 = u >> 16;
    const uint32_t w1 = __builtin_amdgcn_ubfe(u, 16 - D, 16);
    const uint32_t s0 = __builtin_amdgcn_sad_u8(mul1_hash(w0), 0u, 0x64006400u);   // low half: 0x6400 + bytesum
    const uint32_t s = __builtin_amdgcn_sad_hi_u8(mul1_hash(w1), 0u, s0);           // high half: the second's
    h2 hv = __builtin_bit_cast(h2, s);  // bitcast-ok: a whole u32 to f16x2, no lane extract
    hv = __builtin_elementwise_fma(hv, __builtin_bit_cast(h2, 0x1eee1eeeu), __builtin_bit_cast(h2, 0xc931c931u));  // bitcast-ok: whole-vector constants
    return __builtin_bit_cast(uint32_t, hv);  // bitcast-ok: a whole f16x2 to u32, no lane extract
}

// This lane's 8 values of the tile at `tw` (words in LDS or global memory): k rows 8 h .. 8 h + 7 of column c, as four
// packed pairs (rows 8 h + 2 q, 8 h + 2 q + 1).
template <int K2>
__device__ __forceinline__ void decode_own(const uint32_t* tw, const LaneWin<K2>& w, uint32_t (&own)[4]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) own[q] = decode_pair<LaneWin<K2>::D>(__builtin_amdgcn_alignbit(tw[w.hi[q]], tw[w.lo[q]], w.sh[q]));
}

// The tile's B fragment for column lane & 15: this lane's eight k (8 h .. 8 h + 7) from its own decode, then its
// partner's (lane ^ 16, by v_permlanex16). Each half-wave computes its own output rows from its own copies of A and
// B, so a half's k order is free as long as A follows it: lanes 16-31 hold k 8-15 first, and their A fragments are
// loaded with the two 8-k halves swapped (load16h_k, a_swap = 1), which spares the half selects.
__device__ __forceinline__ f16x16 bfrag_from_own(const uint32_t (&own)[4], int lane) {
    (void)lane;
    uint32_t bw[8];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        bw[q] = own[q];
        bw[4 + q] = __builtin_amdgcn_permlanex16(own[q], own[q], 0x76543210u, 0xfedcba98u, false, false);
    }
    f16x16 b;
    __builtin_memcpy(&b, bw, 32);
    return b;
}

__device__ __forceinline__ f16x16 frag8(const uint4& lo, const uint4& hi) {
    const uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    f16x16 a;
    __builtin_memcpy(&a, w, 32);
    return a;
}

// 16 contiguous halves (32-byte aligned) as two 16-byte loads; zero for a null row
__device__ __forceinline__ void load16h(const _Float16* p, uint4& lo, uint4& hi) {
    if (p) {
        const auto q = gl(reinterpret_cast<const uint4*>(p));
        lo = q[0];
        hi = q[1];
    } else {
        lo = hi = make_uint4(0u, 0u, 0u, 0u);
    }
}

// The same for an A fragment against bfrag_from_own's B: lanes 16-31 (sw = 1) take k 8-15 first
__device__ __forceinline__ void load16h_k(const _Float16* p, int sw, uint4& lo, uint4& hi) {
    if (p) {
        const auto q = gl(reinterpret_cast<const uint4*>(p));
        lo = q[sw];
        hi = q[sw ^ 1];
    } else {
        lo = hi = make_uint4(0u, 0u, 0u, 0u);
    }
}

__device__ __forceinline__ f32x8 wmma(const f16x16& a, const f16x16& b, const f32x8& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ void lds_fence() {
    __builtin_amdgcn_wave_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

// One wave's k tiles [kt0, kt0 + n) of a 128-column strip (eight 16 x 16 tiles a k step, contiguous; steps `step` words
// apart) against a single row (xrow: 16 halves a k tile), without the matrix unit. A k step's words are loaded
// coalesced one step ahead into registers, staged in LDS (st: the wave's, 8 * 4 K2 words), and each tile's windows read
// back from there. A lane's four decoded pairs (k rows 8 h + 2 q, + 1 of column c) meet the row's matching input
// pairs in v_dot2_f32_f16; the two half-waves' partial sums are added at the end. acc[t][0] of lanes 0-15 then holds
// output column 16 t + c (row 0 of WMMA's C layout); the other elements are left as they were (zero).
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a), __builtin_bit_cast(h2, b), c, false);  // bitcast-ok: whole u32 to f16x2
}

template <int K2>
__device__ __forceinline__ void strip_tiles_1row(const uint32_t* tiles, int64_t step, int kt0, int n, const _Float16* xrow,
                                                 uint32_t* st, int lane, f32x8 (&acc)[8]) {
    constexpr int TW = 4 * K2, LW = 8 * TW / 32;
    if (n <= 0) return;
    const LaneWin<K2> win(lane);
    const int h = lane >> 4;
    float s[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) s[t] = 0.f;
    uint32_t pf[LW];
    uint4 xa;
    {
        const uint32_t* p = tiles + (int64_t)kt0 * step;
#pragma unroll
        for (int i = 0; i < LW; ++i) pf[i] = __builtin_nontemporal_load(gl(p) + i * 32 + lane);
        xa = *gl(reinterpret_cast<const uint4*>(xrow + 16 * kt0 + 8 * h));
    }
    for (int j = 0; j < n; ++j) {
#pragma unroll
        for (int i = 0; i < LW; ++i) st[i * 32 + lane] = pf[i];
        const uint4 x = xa;
        if (j + 1 < n) {
            const uint32_t* p = tiles + (int64_t)(kt0 + j + 1) * step;
#pragma unroll
            for (int i = 0; i < LW; ++i) pf[i] = __builtin_nontemporal_load(gl(p) + i * 32 + lane);
            xa = *gl(reinterpret_cast<const uint4*>(xrow + 16 * (kt0 + j + 1) + 8 * h));
        }
        lds_fence();
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            uint32_t own[4];
            decode_own<K2>(st + t * TW, win, own);
            float a = dot2(own[0], x.x, s[t]);
            a = dot2(own[1], x.y, a);
            a = dot2(own[2], x.z, a);
            s[t] = dot2(own[3], x.w, a);
        }
        __builtin_amdgcn_wave_barrier();
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float o = __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(s[t]), __float_as_uint(s[t]), 0x76543210u, 0xfedcba98u, false, false));
        acc[t][0] += s[t] + o;
    }
}

// strip_tiles for 2 to 16 rows in strip_tiles_1row's order, so that a row's sums are the same bits whatever rows share
// the launch (decode / verify windows of any width): every row's own v_dot2_f32_f16 chain over the wave's k range,
// then the two half-waves' sums added, with each tile decoded once for all the rows. A part takes TT of the strip's 8
// tiles (t0 ..) for up to RM rows: up to 8 rows all 8 tiles at once, more rows in two parts of 4 tiles (the
// accumulators then fit the registers; each part loads only its tiles' words). A k step's input rows (16 bytes a
// lane: row lane / 2, k half lane % 2, from xlane; nothing past row R - 1) are loaded one step ahead and staged in xst
// (32 uint4, the wave's own); each row's half is then read back as a broadcast. acc as WMMA's C layout: lane l,
// float i -> row 2 i + (l >> 4), column 16 t + (l & 15).
template <int K2, int RM, int TT, int t0>
__device__ __forceinline__ void strip_rows_part(const uint32_t* tiles, int64_t step, int kt0, int n, const _Float16* xlane,
                                                int R, uint32_t* st, uint4* xst, int lane, f32x8 (&acc)[8]) {
    constexpr int TW = 4 * K2, PW = TT * TW, RL = (PW + 31) / 32;
    const LaneWin<K2> win(lane);
    const int h = lane >> 4;
    const bool xlive = (lane >> 1) < R;
    const uint4* xl = gl(reinterpret_cast<const uint4*>(xlane + 8 * (lane & 1)));
    const uint32_t* tp = tiles + t0 * TW;
    float s[RM][TT];
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int t = 0; t < TT; ++t) s[r][t] = 0.f;
    uint32_t pf[RL];
    uint4 xp = make_uint4(0u, 0u, 0u, 0u);
#define TF_DS_EXL3_ROWS_LOAD(kt_)                                                                                       \
    {                                                                                                                   \
        const uint32_t* p_ = tp + (int64_t)(kt_) * step;                                                                \
        _Pragma("unroll") for (int i = 0; i < RL; ++i) {                                                               \
            const int wi = i * 32 + lane;                                                                               \
            pf[i] = wi < PW ? __builtin_nontemporal_load(gl(p_) + wi) : 0u;                                             \
        }                                                                                                               \
        if (xlive) xp = xl[2 * (kt_)];                                                                                  \
    }
    TF_DS_EXL3_ROWS_LOAD(kt0)
    for (int j = 0; j < n; ++j) {
#pragma unroll
        for (int i = 0; i < RL; ++i) {
            const int wi = i * 32 + lane;
            if (wi < PW) st[wi] = pf[i];
        }
        xst[lane] = xp;
        if (j + 1 < n) TF_DS_EXL3_ROWS_LOAD(kt0 + j + 1)
        lds_fence();
        uint32_t own[TT][4];
#pragma unroll
        for (int t = 0; t < TT; ++t) decode_own<K2>(st + t * TW, win, own[t]);
#pragma unroll
        for (int r = 0; r < RM; ++r) {
            if (r < R) {
                const uint4 x = xst[2 * r + h];
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    float a = dot2(own[t][0], x.x, s[r][t]);
                    a = dot2(own[t][1], x.y, a);
                    a = dot2(own[t][2], x.z, a);
                    s[r][t] = dot2(own[t][3], x.w, a);
                }
            }
        }
        __builtin_amdgcn_wave_barrier();
    }
#undef TF_DS_EXL3_ROWS_LOAD
    // this lane's row 2 i + h of each pair: its own half's sum plus the partner's, as strip_tiles_1row adds them (rows
    // past R - 1 are dropped later); whole vectors, so the accumulators stay in registers
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (2 * i + 1 < RM) {
                const float s0 = s[2 * i][t], s1 = s[2 * i + 1][t];
                const float o0 = __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(s0), __float_as_uint(s0), 0x76543210u, 0xfedcba98u, false, false));
                const float o1 = __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(s1), __float_as_uint(s1), 0x76543210u, 0xfedcba98u, false, false));
                v[i] = h ? s1 + o1 : s0 + o0;
            } else {
                v[i] = 0.f;
            }
        }
        acc[t0 + t] += (f32x8){v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
    }
}

template <int K2>
__device__ __forceinline__ void strip_tiles_rows(const uint32_t* tiles, int64_t step, int kt0, int n, const _Float16* xlane,
                                                 int R, uint32_t* st, uint4* xst, int lane, f32x8 (&acc)[8]) {
    if (n <= 0) return;
    if (R <= 8) {
        strip_rows_part<K2, 8, 8, 0>(tiles, step, kt0, n, xlane, R, st, xst, lane, acc);
    } else {
        strip_rows_part<K2, 16, 4, 0>(tiles, step, kt0, n, xlane, R, st, xst, lane, acc);
        __builtin_amdgcn_wave_barrier();   // the second part restages st and xst
        strip_rows_part<K2, 16, 4, 4>(tiles, step, kt0, n, xlane, R, st, xst, lane, acc);
    }
}

// ==== dense linears: exl3.rot_many, exl3.glinear (exl3_linear.zig), exl3.rot_in, exl3.unpack (exl3_prefill.zig) =====

// rot_out: the finished outputs of one row's 128 columns as stored in y, rotated into the next layer's input rows (what
// rot_many computes from y: y's rounding, then rot128_in with the next layer's suh)
__device__ __forceinline__ void rot_out(const tf_ds_exl3_glayer& L, const float (&v)[4], int y_dtype, int lane, size_t row, int col0) {
    float u[4], s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        u[j] = y_dtype == BF16 ? bf16r(v[j]) : y_dtype == F16 ? __half2float(f2h(v[j])) : v[j];
    const int c = L.roff + col0 + 4 * lane;
    load4(reinterpret_cast<const void*>(L.rsuh), F16, c, s);
    rot128_in(u, s, lane);
    store4(reinterpret_cast<void*>(L.rxh), F16, row * (size_t)L.ldr + c, u);
}

// rope_heads (forward) of one row's 128 stored columns when they hold a head's rotary part: v rounded as stored (bf16),
// then re = fma(xe, cs, -(xo sn)), im = fma(xe, sn, xo cs) on each adjacent pair
__device__ __forceinline__ void rope_out(const tf_ds_exl3_glayer& L, float (&v)[4], int lane, size_t row, int col0) {
    const int d0 = (col0 % L.rhd) + 4 * lane, lo = L.rhd - L.rrd;
    if (d0 < lo) return;
    const int64_t p = reinterpret_cast<const int64_t*>(L.rpos)[row];
    const float* rcos = reinterpret_cast<const float*>(L.rcos);
    const float* rsin = reinterpret_cast<const float*>(L.rsin);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int j = (d0 + 2 * q - lo) >> 1;
        const float cs = rcos[p * (L.rrd >> 1) + j], sn = rsin[p * (L.rrd >> 1) + j];
        const float xe = bf16r(v[2 * q]), xo = bf16r(v[2 * q + 1]);
        v[2 * q] = fmaf(xe, cs, -(xo * sn));
        v[2 * q + 1] = fmaf(xe, sn, xo * cs);
    }
}

__device__ __forceinline__ void finish_store(const tf_ds_exl3_glayer& L, float (&v)[4], int lane, size_t row, int col0) {
    finish(v, lane, reinterpret_cast<const __half*>(L.svh), reinterpret_cast<const __half*>(L.bias), col0 + 4 * lane);
    if (L.rcos) rope_out(L, v, lane, row, col0);
    store4(reinterpret_cast<void*>(L.y), L.y_dtype, row * (size_t)L.ldy + col0 + 4 * lane, v);
    if (L.rsuh) rot_out(L, v, L.y_dtype, lane, row, col0);
}

// glinear's blocks for EXL3_WIDE_MIN rows or more: the block's waves split its 128 columns instead of K (8 / WK tiles each),
// so each wave decodes its tiles once a k tile and applies them to every 16-row pass (up to 8) from registers, over
// the split's whole K range. Then per pass: SK == 1, the waves' column tiles meet in LDS and each row's 128 columns
// are finished; SK > 1, the partials go to Z and the block completing the column block (counters[nb], left zero)
// sums the splits in order and finishes every row.
// From 64 rows by default (EXL3_WIDE=0: never). Measured on node1 (w-exl3 harness, real tensors, two runs in either
// order), the layer groups' summed time against the 16-row passes: 0.78-0.93 at 64 rows, 0.83-0.84 at 128, but
// 1.03-1.15 at 17-32 (the wide blocks also hold a higher sclk under the power limit).
#ifndef EXL3_WIDE_MIN
#define EXL3_WIDE_MIN 64
#endif
template <int K2, int WK>
__device__ __forceinline__ void glinear_wide(const tf_ds_exl3_gargs& args, const tf_ds_exl3_glayer& L, float* red, uint32_t* st,
                                             int* last_sh) {
    constexpr int TW = 4 * K2, TPW = 8 / WK, RL = (TPW * TW + 31) / 32;
    const int M = args.M, NP = (M + 15) / 16;
    const _Float16* xh = reinterpret_cast<const _Float16*>(L.xh);
    float* Z = reinterpret_cast<float*>(L.Z);
    int* counters = reinterpret_cast<int*>(L.counters);
    const int K = L.K, N = L.N, SK = L.SK, NB = N >> 7;
    const int b = (int)blockIdx.x - L.first;
    const int nb = b % NB, split = b / NB;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, h = lane >> 4;
    const int per_split = (K >> 4) / SK, kt0 = split * per_split;
    const int t0 = wave * TPW;
    const uint32_t* tiles = reinterpret_cast<const uint32_t*>(L.T) + nb * L.stride_nb + t0 * TW;
    const int col0 = nb * 128;
    const LaneWin<K2> win(lane);
    f32x8 acc[8][TPW];
#pragma unroll
    for (int p = 0; p < 8; ++p)
#pragma unroll
        for (int j = 0; j < TPW; ++j) acc[p][j] = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    // this lane's A row of each pass (rows past M read the last row; their outputs are dropped)
    const _Float16* xrow[8];
#pragma unroll
    for (int p = 0; p < 8; ++p) xrow[p] = xh + (size_t)min(16 * p + (lane & 15), M - 1) * L.ldx;
    uint32_t pf[RL];
#define TF_DS_EXL3_WIDE_LOAD(kt_)                                                                                       \
    {                                                                                                                   \
        const uint32_t* q_ = tiles + (int64_t)(kt_) * L.stride_k;                                                      \
        _Pragma("unroll") for (int i = 0; i < RL; ++i) {                                                               \
            const int wi = i * 32 + lane;                                                                               \
            pf[i] = wi < TPW * TW ? __builtin_nontemporal_load(gl(q_) + wi) : 0u;                                          \
        }                                                                                                               \
    }
    TF_DS_EXL3_WIDE_LOAD(kt0)
    for (int i = 0; i < per_split; ++i) {
        const int kt = kt0 + i;
#pragma unroll
        for (int r = 0; r < RL; ++r) {
            const int wi = r * 32 + lane;
            if (wi < TPW * TW) st[wi] = pf[r];
        }
        if (i + 1 < per_split) TF_DS_EXL3_WIDE_LOAD(kt + 1)
        lds_fence();
        f16x16 B[TPW];
#pragma unroll
        for (int j = 0; j < TPW; ++j) {
            uint32_t own[4];
            decode_own<K2>(st + j * TW, win, own);
            B[j] = bfrag_from_own(own, lane);
        }
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            if (p < NP) {
                uint4 alo, ahi;
                load16h_k(xrow[p] + 16 * kt, h, alo, ahi);
                const f16x16 A = frag8(alo, ahi);
#pragma unroll
                for (int j = 0; j < TPW; ++j) acc[p][j] = wmma(A, B[j], acc[p][j]);
            }
        }
    }
#undef TF_DS_EXL3_WIDE_LOAD
    if (SK == 1) {
        // per pass: the C tiles of rows 16 p .. + 15 to red [16][128], then each row's 128 columns finished
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            if (p >= NP) break;
            const int R = min(16, M - 16 * p);
            __syncthreads();
#pragma unroll
            for (int j = 0; j < TPW; ++j)
#pragma unroll
                for (int r = 0; r < 8; ++r) red[(2 * r + h) * 128 + (t0 + j) * 16 + (lane & 15)] = acc[p][j][r];
            __syncthreads();
            for (int r = wave; r < R; r += WK) {
                float v[4];
                const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
                finish_store(L, v, lane, (size_t)(16 * p + r), col0);
            }
        }
        return;
    }
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        if (p >= NP) break;
#pragma unroll
        for (int j = 0; j < TPW; ++j)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int row = 16 * p + 2 * r + h;
                if (row < M) Z[((size_t)split * M + row) * N + col0 + (t0 + j) * 16 + (lane & 15)] = acc[p][j][r];
            }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) *last_sh = atomicAdd(counters + nb, 1) == SK - 1;
    __syncthreads();
    if (!*last_sh) return;
    __threadfence();
    for (int r = wave; r < M; r += WK) {
        const size_t at = (size_t)r * N + col0 + 4 * lane;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = ld_cg(Z + at + j);
        for (int q = 1; q < SK; ++q)
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] += ld_cg(Z + (size_t)q * M * N + at + j);
        finish_store(L, v, lane, (size_t)r, col0);
    }
    if (threadIdx.x == 0) counters[nb] = 0;                     // every program of the block has arrived
}

// glinear for a block of any layer of the group (linear_grouped.cu glinear_kernel): blocks first_i ..
// first_i + N_i / 128 * SK_i - 1 are layer i's, block b of them the 128-column block b % (N_i / 128) of K split
// b / (N_i / 128); wave w of the WK takes k tiles split * per_warp * WK + w * per_warp .. + per_warp - 1. Rows run in
// passes of 16; the waves' sums are added in wave order (rows 0-7, then 8-15) and, with SK > 1, the splits' partials
// (Z [SK, M, N]) in split order by the block that completes the (pass, column block), which leaves its counter zero.
template <int K2, int WK>
__device__ __forceinline__ void glinear_body(const tf_ds_exl3_gargs& args) {
    constexpr int TW = 4 * K2;
    extern __shared__ __attribute__((aligned(16))) float red[];   // WK * min(M, 8) * 128 floats (the launcher's)
    __shared__ __attribute__((aligned(16))) uint32_t stage[WK][8 * TW];
    __shared__ int last_sh;
    const int M = args.M, RH = M < 8 ? M : 8;
    int li = 0;
#pragma unroll
    for (int q = 1; q < TF_DS_EXL3_GMAX; ++q)
        if (q < args.n && (int)blockIdx.x >= args.l[q].first) li = q;
    const tf_ds_exl3_glayer& L = args.l[li];
    const _Float16* xh = reinterpret_cast<const _Float16*>(L.xh);
    float* Z = reinterpret_cast<float*>(L.Z);
    int* counters = reinterpret_cast<int*>(L.counters);
    const int K = L.K, N = L.N, SK = L.SK;
    const int NB = N >> 7;
    const int b = (int)blockIdx.x - L.first;
    const int nb = b % NB, split = b / NB;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + wave * per_warp;
    const uint32_t* tiles = reinterpret_cast<const uint32_t*>(L.T) + nb * L.stride_nb;
    const int col0 = nb * 128;
#if !defined(EXL3_WIDE) || EXL3_WIDE
    if (M >= EXL3_WIDE_MIN) {
        glinear_wide<K2, WK>(args, L, red, stage[wave], &last_sh);
        return;
    }
#endif

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = min(16, M - m0);
        f32x8 acc[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) acc[t] = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        // every row in the 1-row order (row invariance: a row's outputs do not depend on M); the rows' staging is the
        // wave's first 128 floats of red, free until the pass's reduction (its barrier)
        if (R == 1) {
            strip_tiles_1row<K2>(tiles, L.stride_k, kt0, per_warp, xh + (size_t)m0 * L.ldx, stage[wave], lane, acc);
        } else {
            const int ar = m0 + min(lane >> 1, R - 1);
            strip_tiles_rows<K2>(tiles, L.stride_k, kt0, per_warp, xh + (size_t)ar * L.ldx, R, stage[wave],
                                 reinterpret_cast<uint4*>(red + wave * 128), lane, acc);
        }

#pragma unroll
        for (int hp = 0; hp < 2; ++hp) {
            const int rlo = 8 * hp;
            if (rlo >= R) break;
            const int rn = min(R - rlo, 8);
            __syncthreads();                                    // red is reused by every half and pass
#pragma unroll
            for (int t = 0; t < 8; ++t)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int rr = 2 * i + (lane >> 4);
                    if (rr < RH) red[(wave * RH + rr) * 128 + t * 16 + (lane & 15)] = acc[t][4 * hp + i];
                }
            __syncthreads();
            if (SK == 1) {
                for (int r = wave; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish_store(L, v, lane, (size_t)(m0 + rlo + r), col0);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last_sh = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last_sh) {
                __threadfence();
                for (int r = wave; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float v[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) v[j] = ld_cg(Z + at + j);
                    for (int q = 1; q < SK; ++q)
#pragma unroll
                        for (int j = 0; j < 4; ++j) v[j] += ld_cg(Z + (size_t)q * M * N + at + j);
                    finish_store(L, v, lane, (size_t)m0 + r, col0);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                                        // red is reused in the next pass
    }
}

}  // namespace

// rot_in for every layer of a group (linear_grouped.cu rot_many_kernel): a wave a (layer, row, 128-wide block).
extern "C" __global__ void __launch_bounds__(128) tf_ds_exl3_rot_many(tf_ds_exl3_rargs args) {
    const int w = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    int li = 0;
#pragma unroll
    for (int q = 1; q < TF_DS_EXL3_GMAX; ++q)
        if (q < args.n && w >= args.l[q].first) li = q;
    const tf_ds_exl3_rlayer& L = args.l[li];
    const int nblk = L.K >> 7, local = w - L.first;
    if (local >= args.M * nblk) return;                       // past the last layer's rows (whole waves)
    const int row = local / nblk, k = (local % nblk) * 128 + 4 * lane;
    float v[4], s[4];
    load4(reinterpret_cast<const void*>(L.x), L.x_dtype, (size_t)row * L.ldx + k, v);
    load4(reinterpret_cast<const void*>(L.suh), F16, k, s);
    rot128_in(v, s, lane);
    store4(reinterpret_cast<void*>(L.xh), F16, (size_t)row * L.K + k, v);
}

#define TF_DS_EXL3_GLINEAR(K2_, WK_)                                                                                    \
    extern "C" __global__ void __launch_bounds__(WK_ * 32) tf_ds_exl3_glinear_k##K2_##_w##WK_(tf_ds_exl3_gargs args) { \
        glinear_body<K2_, WK_>(args);                                                                                   \
    }
TF_DS_EXL3_GLINEAR(4, 4)
TF_DS_EXL3_GLINEAR(4, 8)
TF_DS_EXL3_GLINEAR(6, 4)
TF_DS_EXL3_GLINEAR(6, 8)
TF_DS_EXL3_GLINEAR(8, 4)
TF_DS_EXL3_GLINEAR(8, 8)
TF_DS_EXL3_GLINEAR(10, 4)
TF_DS_EXL3_GLINEAR(10, 8)
TF_DS_EXL3_GLINEAR(12, 4)
TF_DS_EXL3_GLINEAR(12, 8)
TF_DS_EXL3_GLINEAR(16, 4)
TF_DS_EXL3_GLINEAR(16, 8)
#undef TF_DS_EXL3_GLINEAR

// The prompt path's input rotation (linear.cu rot_in_kernel): xh [M, K] fp16 (contiguous) from x's rows (row stride
// ldx elements): a wave a 128-column block, four blocks a program, a row a grid row.
extern "C" __global__ void __launch_bounds__(128) tf_ds_exl3_rot_in(const void* x, int32_t x_dtype, int64_t ldx,
                                                                   const __half* suh, __half* xh, int32_t K) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(x, x_dtype, (size_t)row * (size_t)ldx + k, v);
    load4(suh, F16, k, s);
    rot128_in(v, s, lane);
    store4(xh, F16, (size_t)row * K + k, v);
}

// W_q [K, N] fp16 from a layer's words in the strips layout (linear.cu unpack_kernel): a wave a 16 x 16 tile.
namespace {
// The decoded tile goes through LDS (a lane holds a column's eight k) so that each lane stores 16 contiguous bytes of a
// row (row lane / 2, columns 8 (lane % 2) .. + 7): one 16-byte store a lane instead of eight 2-byte ones.
template <int K2>
__device__ __forceinline__ void unpack_body(const uint32_t* T, uint16_t* W, int N, int64_t stride_k, int64_t stride_nb) {
    __shared__ __attribute__((aligned(16))) uint16_t t[16][16];
    const int kt = blockIdx.y, nt = blockIdx.x, lane = threadIdx.x;
    const uint32_t* tile = T + kt * stride_k + (nt >> 3) * stride_nb + (nt & 7) * (4 * K2);
    const LaneWin<K2> win(lane);
    uint32_t own[4];
    decode_own<K2>(tile, win, own);
    const int c = lane & 15, h = lane >> 4;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        t[8 * h + 2 * q][c] = (uint16_t)(own[q] & 0xffffu);
        t[8 * h + 2 * q + 1][c] = (uint16_t)(own[q] >> 16);
    }
    lds_fence();
    const int r = lane >> 1, half = lane & 1;
    *reinterpret_cast<uint4*>(W + (size_t)(kt * 16 + r) * N + nt * 16 + 8 * half) = *reinterpret_cast<const uint4*>(&t[r][8 * half]);
}
}  // namespace

#define TF_DS_EXL3_UNPACK(K2_)                                                                                          \
    extern "C" __global__ void __launch_bounds__(32) tf_ds_exl3_unpack_k##K2_(const uint32_t* T, uint16_t* W, int32_t N, \
                                                                             int64_t stride_k, int64_t stride_nb) {     \
        unpack_body<K2_>(T, W, N, stride_k, stride_nb);                                                                 \
    }
TF_DS_EXL3_UNPACK(4)
TF_DS_EXL3_UNPACK(6)
TF_DS_EXL3_UNPACK(8)
TF_DS_EXL3_UNPACK(10)
TF_DS_EXL3_UNPACK(12)
TF_DS_EXL3_UNPACK(16)
#undef TF_DS_EXL3_UNPACK

// ==== the dense prompt GEMM: exl3.gemm (exl3_prefill.zig matmul on HIP) ==========================================
// out [M, N] (row stride o_stride, bf16 or fp32) = H128(xh @ W_q) / sqrt(128) * svh for the rotated rows xh [M, K]
// (rot_in's) and a layer's words in the strips layout: the trellis decoded inside the GEMM, so W_q is never written
// (exl3.unpack + tri._gemm's work in one launch). The body is w-pf's pf_gemm (node0 ~/tfp/pf 867e6f5, after
// w-exl3's w_gemm): a block is BM rows x one 128-column Hadamard strip over the whole K (no split); 8 waves (4 rows x
// 2 columns of the block), each decoding one of the strip's 8 tiles per k tile into LDS ([k half][column][8] halves,
// bank-conflict free) from raw words staged in its own LDS slot, the decode of stage s + 1 split around the WMMAs
// of stage s, raw words PFS stages ahead in registers, the A rows double-buffered in LDS. BM = 256 with one k tile a
// stage (16 accumulators a wave) where the grid keeps >= 120 blocks, else BM = 128; two k tiles a stage either way
// (BM = 256 with one: wo_b 20%, wq_b 6% slower at the same bits; p-pre ~/tfp/ppre/bench/gemmbench). Changes here: strips
// addressing (stride_k, stride_nb), global loads, bf16 / fp32 output with finish()'s epilogue order (H, *
// 1/sqrt(128), * svh), even K2 only (every dense linear of the checkpoint).
namespace {

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

// the tile window of a pair: words j, j + 1 of the staged tile (a wrap word after its last), alignbit shift sh
template <int K>
struct GemmLaneDec {
    int j[4], sh[4];
    __device__ __forceinline__ explicit GemmLaneDec(int lane) {
        constexpr int TB = 256 * K;
        const int col = lane & 15, h = lane >> 4;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int p0 = 32 * (col & 7) + 8 * q + 4 * (col >> 3) + 2 * h;
            int b = ((p0 + 1) * K - 16 + TB) % TB;
            if (b == 0) b = TB;
            j[q] = (b - 1) >> 5;
            sh[q] = 32 - (b - 32 * j[q]);
        }
    }
};

// one tile's raw words, lane-strided
template <int K>
struct GemmRaw {
    static constexpr int TW = 8 * K, NL = (TW + 31) / 32;
    uint32_t w[NL];
    __device__ __forceinline__ void load(const uint32_t* p, int lane) {
#pragma unroll
        for (int i = 0; i < NL; ++i) {
            const int idx = i * 32 + lane;
            w[i] = idx < TW ? __builtin_nontemporal_load(gl(p) + idx) : 0u;
        }
    }
    __device__ __forceinline__ void stage(uint32_t* st, int lane) const {   // st[0..TW), st[TW] = st[0]
#pragma unroll
        for (int i = 0; i < NL; ++i) {
            const int idx = i * 32 + lane;
            if (idx < TW) {
                st[idx] = w[i];
                if (idx == 0) st[TW] = w[i];
            }
        }
    }
};

typedef _Float16 GemmBTile[2][16][8];

template <int K>
struct GemmWords {
    uint32_t w[4][2];
    __device__ __forceinline__ void read(const uint32_t* st, const GemmLaneDec<K>& d) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            w[q][0] = st[d.j[q]];
            w[q][1] = st[d.j[q] + 1];
        }
    }
    __device__ __forceinline__ void decode_store(const GemmLaneDec<K>& d, GemmBTile& bt, int lane) const {
        uint32_t v[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = decode_pair<K>(__builtin_amdgcn_alignbit(w[q][0], w[q][1], d.sh[q]));
        *reinterpret_cast<u32x4*>(&bt[lane >> 4][lane & 15][0]) = (u32x4){v[0], v[1], v[2], v[3]};
    }
};

__device__ __forceinline__ f16x16 gemm_load_b(const GemmBTile& bt, int col) {
    const u32x4 lo = *reinterpret_cast<const u32x4*>(&bt[0][col][0]);
    const u32x4 hi = *reinterpret_cast<const u32x4*>(&bt[1][col][0]);
    const uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    f16x16 b;
    __builtin_memcpy(&b, w, 32);
    return b;
}

__device__ __forceinline__ f16x16 gemm_load_a(const _Float16* row16) {
    const uint4* p = reinterpret_cast<const uint4*>(row16);
    return frag8(p[0], p[1]);
}

template <int K2, int BM, int BK>
__device__ __forceinline__ void gemm_body(const _Float16* xh, const uint32_t* words, int64_t stride_k, int64_t stride_nb,
                                          const __half* svh, void* out, int out_f32, int64_t o_stride, int M, int Kd, int N) {
    static_assert((K2 & 1) == 0, "exl3.gemm: even K2");
    constexpr int K = K2 / 2, TW = 8 * K;
    constexpr int THREADS = 256;
    constexpr int RBW = BM / 64;                     // row blocks per wave (waves 4 rows x 2 columns)
    constexpr int AROW = BK * 16 + 8;                // halves per staged A row (8 halves of bank padding)
    constexpr int SEGS = BM * BK * 2, SPT = SEGS / THREADS;   // 16-byte A segments per stage
    constexpr int PFS = BK == 1 ? 4 : 2;             // stages of raw words in flight (4 k tiles)
    constexpr int CSW = 132;                         // epilogue floats per row
    constexpr int AS_B = 2 * BM * AROW * 2, BS_B = 2 * BK * 8 * (int)sizeof(GemmBTile), CS_B = 32 * CSW * 4;
    static_assert(SEGS % THREADS == 0, "A segments");
    __shared__ __attribute__((aligned(16))) unsigned char smem[AS_B + BS_B > CS_B ? AS_B + BS_B : CS_B];
    __shared__ __attribute__((aligned(16))) uint32_t sts[8][BK][TW + 1];
    _Float16 (*As)[BM][AROW] = reinterpret_cast<_Float16 (*)[BM][AROW]>(smem);
    GemmBTile (*Bs)[BK][8] = reinterpret_cast<GemmBTile (*)[BK][8]>(smem + AS_B);
    float* cs = reinterpret_cast<float*>(smem);

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int strips = N / 128;
    const int strip = (int)blockIdx.x % strips;
    const int m0 = ((int)blockIdx.x / strips) * BM;
    const int KT = Kd / 16, NS = KT / BK;
    const int wr = wave >> 1, wc = wave & 1;
    const uint32_t* tbase = words + strip * stride_nb + wave * TW;   // this wave's tile of the strip, k tile 0
    const GemmLaneDec<K> ld(lane);
    uint32_t (*st)[TW + 1] = sts[wave];

    u32x4 areg[SPT];
#define TF_DS_EXL3_A_LOAD(s_)                                                                                           \
    _Pragma("unroll") for (int i = 0; i < SPT; ++i) {                                                                   \
        const int sg_ = tid + i * THREADS, r_ = sg_ / (BK * 2), q_ = sg_ % (BK * 2), row_ = m0 + r_;                     \
        areg[i] = row_ < M ? *gl(reinterpret_cast<const u32x4*>(xh + (size_t)row_ * Kd + (s_) * BK * 16 + q_ * 8))       \
                           : (u32x4){0u, 0u, 0u, 0u};                                                                   \
    }
#define TF_DS_EXL3_A_STORE(buf_)                                                                                        \
    _Pragma("unroll") for (int i = 0; i < SPT; ++i) {                                                                   \
        const int sg_ = tid + i * THREADS, r_ = sg_ / (BK * 2), q_ = sg_ % (BK * 2);                                     \
        *reinterpret_cast<u32x4*>(&As[buf_][r_][q_ * 8]) = areg[i];                                                     \
    }

    f32x8 acc[RBW][4];
#pragma unroll
    for (int i = 0; i < RBW; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // raw ring: ring[u] holds stage s + 1 at stage s = s0 + u (u = s % PFS); stage 0 decoded here
    GemmRaw<K> ring[PFS][BK];
    {
        GemmRaw<K> r0[BK];
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) r0[kk].load(tbase + (int64_t)kk * stride_k, lane);
#pragma unroll
        for (int u = 0; u < PFS; ++u)
#pragma unroll
            for (int kk = 0; kk < BK; ++kk)
                if (u + 1 < NS) ring[u][kk].load(tbase + (int64_t)((u + 1) * BK + kk) * stride_k, lane);
        TF_DS_EXL3_A_LOAD(0)
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) r0[kk].stage(st[kk], lane);
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            GemmWords<K> w;
            w.read(st[kk], ld);
            w.decode_store(ld, Bs[0][kk][wave], lane);
        }
        TF_DS_EXL3_A_STORE(0)
    }
    __syncthreads();

    for (int s0 = 0; s0 < NS; s0 += PFS) {
#pragma unroll
        for (int u = 0; u < PFS; ++u) {
            const int s = s0 + u;
            if (s >= NS) break;
            const int buf = s & 1;
            const bool next = s + 1 < NS;
            if (next) {
                TF_DS_EXL3_A_LOAD(s + 1)
#pragma unroll
                for (int kk = 0; kk < BK; ++kk) {
                    ring[u][kk].stage(st[kk], lane);
                    if (s + 1 + PFS < NS) ring[u][kk].load(tbase + (int64_t)((s + 1 + PFS) * BK + kk) * stride_k, lane);
                }
            }
            __builtin_amdgcn_wave_barrier();
            GemmWords<K> wn[BK];
            if (next) {
#pragma unroll
                for (int kk = 0; kk < BK; ++kk) wn[kk].read(st[kk], ld);
            }
#pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                f16x16 B[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) B[j] = gemm_load_b(Bs[buf][kk][wc * 4 + j], lane & 15);
#pragma unroll
                for (int i = 0; i < RBW; ++i) {
                    const f16x16 A = gemm_load_a(&As[buf][wr * (RBW * 16) + i * 16 + (lane & 15)][kk * 16]);
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] = wmma(A, B[j], acc[i][j]);
                }
            }
            if (next) {
#pragma unroll
                for (int kk = 0; kk < BK; ++kk) wn[kk].decode_store(ld, Bs[buf ^ 1][kk][wave], lane);
                TF_DS_EXL3_A_STORE(buf ^ 1)
            }
            __syncthreads();
        }
    }
#undef TF_DS_EXL3_A_LOAD
#undef TF_DS_EXL3_A_STORE

    // epilogue in passes of 32 rows: C -> LDS (fp32), each row's 128 columns through finish (H, 1/sqrt(128), svh)
    constexpr int PPW = RBW / 2;                     // passes per wave row group
#pragma unroll
    for (int pass = 0; pass < 4 * PPW; ++pass) {
        if (wr == pass / PPW) {
            const int i0 = (pass % PPW) * 2;
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int r = 0; r < 8; ++r)
                        cs[(ii * 16 + 2 * r + (lane >> 4)) * CSW + (wc * 4 + j) * 16 + (lane & 15)] = acc[i0 + ii][j][r];
        }
        __syncthreads();
        for (int rr = wave; rr < 32; rr += 8) {
            const int row = m0 + pass * 32 + rr;
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = cs[rr * CSW + lane * 4 + i];
            finish(v, lane, svh, nullptr, strip * 128 + lane * 4);
            if (row < M) store4(out, out_f32 ? F32 : BF16, (size_t)row * o_stride + strip * 128 + lane * 4, v);
        }
        __syncthreads();
    }
}

}  // namespace

// exl3.gemm {k2, cb 2, bm 128 | 256}: grid (N / 128) * ceil(M / bm), block 256, no dynamic LDS
#define TF_DS_EXL3_GEMM(K2_, BM_, BK_)                                                                                  \
    extern "C" __global__ void __launch_bounds__(256) tf_ds_exl3_gemm_k##K2_##_m##BM_(                                  \
        const _Float16* xh, const uint32_t* words, int64_t stride_k, int64_t stride_nb, const __half* svh, void* out,  \
        int64_t o_stride, int32_t m, int32_t k, int32_t n, int32_t out_f32) {                                          \
        gemm_body<K2_, BM_, BK_>(xh, words, stride_k, stride_nb, svh, out, out_f32, o_stride, m, k, n);                \
    }
TF_DS_EXL3_GEMM(8, 128, 2)
TF_DS_EXL3_GEMM(8, 256, 2)
TF_DS_EXL3_GEMM(10, 128, 2)
TF_DS_EXL3_GEMM(10, 256, 2)
TF_DS_EXL3_GEMM(12, 128, 2)
TF_DS_EXL3_GEMM(12, 256, 2)
TF_DS_EXL3_GEMM(16, 128, 2)
TF_DS_EXL3_GEMM(16, 256, 2)
#undef TF_DS_EXL3_GEMM

// ==== routed experts (exl3_experts.zig): experts.cu's plain kernels =================================================

namespace {

constexpr int PLACE_THREADS = 256;
constexpr int LIST_THREADS = 1024;
constexpr int LIST_NONE = 0x7fffffff;
constexpr int PREP_THREADS = 256;

__device__ __forceinline__ int block_sum(int v, int* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    int s = 0;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += sh[w];
    return s;
}

// a member code (row * 32 + slot) as its pick index, or -1
__device__ __forceinline__ int member_p(int code, int slots) { return code >= 0 ? (code >> 5) * slots + (code & 31) : -1; }

// rot_in_kernel's program (member row p, 128-block blk, matrix mat) of a bf16 input, one a wave
__device__ __forceinline__ void experts_rot_one(const uint16_t* x, int x_stride, const int* pick, const __half* suh0,
                                                const __half* suh1, __half* out0, __half* out1, int K, int slots, int E,
                                                int p, int blk, int mat, int lane) {
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const __half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const uint16_t* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = bf16_bits_to_f(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    __half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = f2h(v[j] * HAD_SCALE);
}

// gate / up sums (rotated) * svh, SwiGLU (act_mode 0: GLM's bf16 roundings, 1: fp32) * suh_d, rotated: one member row's
// 128 columns of Xd, from its gate and up sums g, u (4 a lane)
__device__ __forceinline__ void gateup_act(float (&gv)[4], float (&uv)[4], int lane, const __half* svh_g, const __half* svh_u,
                                          const __half* suh_d, int e, int N, int n, float limit, int act_mode, __half* o) {
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            const float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            const float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit), limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            const float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
            const float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = f2h(v[j] * HAD_SCALE);
}

}  // namespace

// A block per expert counts its picks (experts.cu group_count_kernel).
extern "C" __global__ void __launch_bounds__(PLACE_THREADS) tf_ds_exl3_experts_group_count(const int32_t* pick, int32_t* counts,
                                                                                           int32_t n) {
    __shared__ int sh[PLACE_THREADS / 32];
    const int e = blockIdx.x;
    int c = 0;
    for (int i = threadIdx.x; i < n; i += PLACE_THREADS) c += pick[i] == e;
    c = block_sum(c, sh);
    if (threadIdx.x == 0) counts[e] = c;
}

// A block per expert: a used expert takes its place in id order and lists its members (row * 32 + slot) in row order,
// -1 after the last; block 0 also writes the used count (experts.cu group_place_kernel).
extern "C" __global__ void __launch_bounds__(PLACE_THREADS) tf_ds_exl3_experts_group_place(const int32_t* pick, const int32_t* counts,
                                                                                           int32_t* uids, int32_t* ucount,
                                                                                           int32_t* members, int32_t n,
                                                                                           int32_t slots, int32_t E,
                                                                                           int32_t maxm) {
    __shared__ int sh[PLACE_THREADS / 32];
    __shared__ int warp_tot[PLACE_THREADS / 32];
    const int e = blockIdx.x;
    int before = 0;
    for (int i = threadIdx.x; i < e; i += PLACE_THREADS) before += counts[i] > 0;
    const int u = block_sum(before, sh);
    if (e == 0) {
        int used = 0;
        for (int i = threadIdx.x; i < E; i += PLACE_THREADS) used += counts[i] > 0;
        used = block_sum(used, sh);
        if (threadIdx.x == 0) ucount[0] = used;
    }
    if (counts[e] == 0) return;
    if (threadIdx.x == 0) uids[u] = e;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int base = 0;
    for (int start = 0; start < n; start += PLACE_THREADS) {
        const int i = start + threadIdx.x;
        const int hit = i < n && pick[i] == e;
        int inc = hit;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up(inc, o, 32);
            if (lane >= o) inc += v;
        }
        __syncthreads();
        if (lane == 31) warp_tot[warp] = inc;
        __syncthreads();
        int prior = 0, total = 0;
        for (int w = 0; w < PLACE_THREADS / 32; ++w) {
            prior += w < warp ? warp_tot[w] : 0;
            total += warp_tot[w];
        }
        const int j = base + prior + inc - hit;
        if (hit && j < maxm) members[u * maxm + j] = (i / slots) * 32 + (i % slots);
        base += total;
    }
    for (int j = base + threadIdx.x; j < maxm; j += PLACE_THREADS) members[u * maxm + j] = -1;
}

// Prompt chunks' work lists (experts.cu work_list_kernel): the (expert place, member group) pairs of the used places in
// place order, then sentinel pairs up to each list's length. One block.
extern "C" __global__ void __launch_bounds__(LIST_THREADS) tf_ds_exl3_experts_work_list(const int32_t* counts, const int32_t* uids,
                                                                                        const int32_t* ucount, int32_t maxu,
                                                                                        int32_t* work0, int32_t rows0, int32_t n0,
                                                                                        int32_t* work1, int32_t rows1, int32_t n1,
                                                                                        int32_t skip_e) {
    __shared__ int warp_tot[LIST_THREADS / 32];
    const int nu = ucount[0];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int l = 0; l < (work1 ? 2 : 1); ++l) {
        int* work = l ? work1 : work0;
        const int rows = l ? rows1 : rows0, n = l ? n1 : n0;
        int base = 0;
        for (int start = 0; start < maxu; start += LIST_THREADS) {
            const int p = start + threadIdx.x;
            const int e = (p < maxu && p < nu) ? uids[p] : -1;
            const int g = (e >= 0 && e != skip_e) ? (counts[e] + rows - 1) / rows : 0;
            int inc = g;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up(inc, o, 32);
                if (lane >= o) inc += v;
            }
            if (lane == 31) warp_tot[warp] = inc;
            __syncthreads();
            int prior = 0, total = 0;
            for (int w = 0; w < LIST_THREADS / 32; ++w) {
                prior += w < warp ? warp_tot[w] : 0;
                total += warp_tot[w];
            }
            const int first = base + prior + inc - g;
            for (int j = 0; j < g; ++j) {
                const int q = first + j;
                if (q < n) {
                    work[2 * q] = p;
                    work[2 * q + 1] = j;
                }
            }
            base += total;
            __syncthreads();
        }
        for (int q = base + threadIdx.x; q < n; q += LIST_THREADS) {
            work[2 * q] = LIST_NONE;
            work[2 * q + 1] = 0;
        }
    }
}

// Program (member row, 128-block of K, matrix): Xh = fp16((x * suh) @ H / sqrt(128)) for gate and up of every routed
// slot (experts.cu rot_in_kernel<bf16>; grid (P, K / 128, 2), a wave a program).
extern "C" __global__ void __launch_bounds__(32) tf_ds_exl3_experts_rot_in(const uint16_t* x, int32_t x_stride, const int32_t* pick,
                                                                           const __half* suh0, const __half* suh1, __half* out0,
                                                                           __half* out1, int32_t K, int32_t slots, int32_t E) {
    experts_rot_one(x, x_stride, pick, suh0, suh1, out0, out1, K, slots, E, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x);
}

// Program (member row, 128-block of the width): gate / up splits summed in order, then the SwiGLU epilogue into Xd
// (experts.cu gateup_epilogue_kernel).
extern "C" __global__ void __launch_bounds__(32) tf_ds_exl3_experts_gateup_epilogue(const float* Z, const int32_t* pick,
                                                                                    const __half* svh_g, const __half* svh_u,
                                                                                    const __half* suh_d, __half* xd, int32_t P,
                                                                                    int32_t N, int32_t SK, int32_t E,
                                                                                    float limit, int32_t act_mode) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    gateup_act(gv, uv, lane, svh_g, svh_u, suh_d, e, N, n, limit, act_mode, xd + (size_t)p * N + n);
}

// Program (row, 128-block of the model width), a wave a slot: each routed slot's output (splits summed in order,
// rotated, * svh_d; to y with store_y), then the slots weighted by wts in slot order (fma chain from 0), + add last
// (experts.cu down_combine_kernel).
extern "C" __global__ void __launch_bounds__(1024) tf_ds_exl3_experts_down_combine(const float* Z, const int32_t* pick,
                                                                                  const __half* svh_d, float* y, const float* wts,
                                                                                  const float* add, float* out, int32_t P,
                                                                                  int32_t D, int32_t SK, int32_t E,
                                                                                  int32_t slots, int32_t store_y) {
    __shared__ float4 part[32][32];
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4];
    if (e >= 0 && e < E) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            if (store_y) y[(size_t)p * D + n + j] = o[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = store_y ? y[(size_t)p * D + n + j] : 0.f;
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) out[(size_t)r * D + n + j] = add ? acc[j] + add[(size_t)r * D + n + j] : acc[j];
}

// Decode windows' first launch (experts.cu decode_prep_kernel<bf16>): block 0 adds one to epoch, groups the picks
// (distinct experts < E in id order, members row * 32 + slot in row order, -1 after the last) and, with wts, combines
// the rows that route no slot; the other blocks rotate gate / up's input rows, a wave a (member row, 128-block,
// matrix). Dynamic LDS: 2 * R * slots ints.
extern "C" __global__ void __launch_bounds__(PREP_THREADS) tf_ds_exl3_experts_decode_prep(
    const uint16_t* x, int32_t x_stride, const int32_t* pick, const __half* suh0, const __half* suh1, __half* out0, __half* out1,
    int32_t K, int32_t slots, int32_t E, int32_t R, int32_t* uids, int32_t* ucount, int32_t* members, int32_t maxm, const float* wts,
    const float* y, const float* add, float* out, int32_t D, int32_t store_y, int32_t* epoch) {
    const int n = R * slots;
    if (blockIdx.x == 0 && threadIdx.x == 0 && epoch != nullptr) epoch[0] += 1;
    if (blockIdx.x == 0) {
        extern __shared__ int sh_prep[];
        __shared__ int used_sh;
        int* sp = sh_prep;
        int* first = sh_prep + n;
        for (int i = threadIdx.x; i < n; i += PREP_THREADS) sp[i] = pick[i];
        if (threadIdx.x == 0) used_sh = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += PREP_THREADS) {
            const int e = sp[i];
            int f = e >= 0 && e < E;
            for (int j = 0; f && j < i; ++j) f = sp[j] != e;
            first[i] = f;
        }
        __syncthreads();
        int used = 0;
        for (int i = threadIdx.x; i < n; i += PREP_THREADS) {
            const int e = sp[i];
            if (e < 0 || e >= E) continue;
            int place = 0, pos = 0, tot = 0;
            for (int j = 0; j < n; ++j) {
                const int ej = sp[j];
                place += first[j] && ej < e;
                tot += ej == e;
                pos += ej == e && j < i;
            }
            if (pos < maxm) members[place * maxm + pos] = (i / slots) * 32 + (i % slots);
            if (first[i]) {
                uids[place] = e;
                for (int k = tot; k < maxm; ++k) members[place * maxm + k] = -1;
                ++used;
            }
        }
        if (used) atomicAdd(&used_sh, used);
        __syncthreads();
        if (threadIdx.x == 0) ucount[0] = used_sh;
        if (wts == nullptr) return;
        for (int r = 0; r < R; ++r) {
            int routed = 0;
            for (int q = 0; q < slots; ++q) routed += sp[r * slots + q] >= 0 && sp[r * slots + q] < E;
            if (routed) continue;
            for (int d = threadIdx.x * 4; d < D; d += PREP_THREADS * 4) {
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                for (int q = 0; q < slots; ++q) {
                    const float w = wts[r * slots + q];
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[j] = fmaf(w, store_y ? y[((size_t)r * slots + q) * D + d + j] : 0.f, acc[j]);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) out[(size_t)r * D + d + j] = add ? acc[j] + add[(size_t)r * D + d + j] : acc[j];
            }
        }
        return;
    }
    const int nblk = K / 128;
    const int item = (blockIdx.x - 1) * (PREP_THREADS / 32) + (threadIdx.x >> 5);
    if (item >= n * nblk * 2) return;
    const int p = item / (nblk * 2), mat = (item / nblk) % 2, blk = item % nblk;
    experts_rot_one(x, x_stride, pick, suh0, suh1, out0, out1, K, slots, E, p, blk, mat, threadIdx.x & 31);
}

// ==== routed experts: the grouped GEMMs (experts_grouped.cuh's ids), any K2 of [LO, HI] per expert ================

namespace {

// a K2 switch over the instance's range: an expert whose K2 lies outside it traps (as the CUDA kernels)
#define TF_DS_EXL3_K2_CASE(K2_, LO_, HI_, CALL)                                                                         \
    case K2_:                                                                                                           \
        if constexpr (K2_ >= LO_ && K2_ <= HI_) { CALL(K2_); }                                                          \
        else __builtin_trap();                                                                                          \
        break;
#define TF_DS_EXL3_K2_SWITCH(k2, LO_, HI_, CALL)                                                                       \
    switch (k2) {                                                                                                       \
        TF_DS_EXL3_K2_CASE(2, LO_, HI_, CALL) TF_DS_EXL3_K2_CASE(3, LO_, HI_, CALL) TF_DS_EXL3_K2_CASE(4, LO_, HI_, CALL)  \
        TF_DS_EXL3_K2_CASE(5, LO_, HI_, CALL) TF_DS_EXL3_K2_CASE(6, LO_, HI_, CALL) TF_DS_EXL3_K2_CASE(7, LO_, HI_, CALL)  \
        TF_DS_EXL3_K2_CASE(8, LO_, HI_, CALL) TF_DS_EXL3_K2_CASE(9, LO_, HI_, CALL) TF_DS_EXL3_K2_CASE(10, LO_, HI_, CALL) \
        TF_DS_EXL3_K2_CASE(11, LO_, HI_, CALL) TF_DS_EXL3_K2_CASE(12, LO_, HI_, CALL)                                   \
        TF_DS_EXL3_K2_CASE(13, LO_, HI_, CALL) TF_DS_EXL3_K2_CASE(14, LO_, HI_, CALL)                                   \
        TF_DS_EXL3_K2_CASE(15, LO_, HI_, CALL) TF_DS_EXL3_K2_CASE(16, LO_, HI_, CALL)                                   \
        default: __builtin_trap();                                                                                      \
    }

// ---- exl3.experts.grouped_mma: prompt gate / up -----------------------------------------------------------------------
// A program per (64 member rows, 64 columns, expert place, matrix) from the work list (grouped_mma_kernel's grid: (1,
// N / 64, pairs x 2)), eight waves. The k tiles go two a step: each wave decodes one of the step's eight 16 x 16 tiles
// (2 k tiles x 4 column tiles) into LDS (column-major fp16, the B fragments' order) while every wave's WMMA read the
// tiles decoded the step before (two buffers, one barrier a step); wave w then multiplies member rows 16 (w % 4) .. +
// 15 by columns 32 (w / 4) .. + 31. Each output is one fp32 chain over K in order: Z [mats, P, N] holds the whole sum
// (the CUDA kernel's FOLD: the splits and K chains summed, one split's worth).
constexpr int MMA_ROWS = 64, MMA_COLS = 64, BCOL = 24;     // BCOL: halves a staged column (16 k + pad)

template <int K2>
__device__ __forceinline__ void mma_tiles(const uint32_t* T, int NTILES, int KT, int nt0, const _Float16* arow, int wave,
                                          int lane, _Float16 (*Bs)[2][4][16][BCOL], uint32_t* raw, f32x8 (&acc)[2]) {
    constexpr int TW = 4 * K2, RL = (TW + 31) / 32;
    const LaneWin<K2> win(lane);
    const int kk = wave >> 2, ntl = wave & 3;               // the tile this wave decodes in a step
    const int cg = wave >> 2;
    const int steps = (KT + 1) / 2;
    uint32_t rw[RL];
#define TF_DS_EXL3_TILE_OF(s_) (T + ((size_t)(2 * (s_) + kk) * NTILES + nt0 + ntl) * TW)
    // step 0's tile into buffer 0, step 1's words into registers
    {
        const bool ok = kk < KT;
        const uint32_t* tp = TF_DS_EXL3_TILE_OF(0);
#pragma unroll
        for (int i = 0; i < RL; ++i) {
            const int wi = i * 32 + lane;
            if (wi < TW) raw[wi] = ok ? __builtin_nontemporal_load(gl(tp) + wi) : 0u;
        }
        lds_fence();
        uint32_t own[4];
        decode_own<K2>(raw, win, own);
        *reinterpret_cast<uint4*>(&Bs[0][kk][ntl][lane & 15][8 * (lane >> 4)]) = make_uint4(own[0], own[1], own[2], own[3]);
        __builtin_amdgcn_wave_barrier();
        if (steps > 1) {
            const bool ok1 = 2 + kk < KT;
            const uint32_t* tp1 = TF_DS_EXL3_TILE_OF(1);
#pragma unroll
            for (int i = 0; i < RL; ++i) {
                const int wi = i * 32 + lane;
                rw[i] = (wi < TW && ok1) ? __builtin_nontemporal_load(gl(tp1) + wi) : 0u;
            }
        }
    }
    uint4 a0lo, a0hi, a1lo, a1hi;
    load16h(arow, a0lo, a0hi);
    load16h(arow && KT > 1 ? arow + 16 : nullptr, a1lo, a1hi);
    for (int s = 0; s < steps; ++s) {
        __syncthreads();                                    // Bs[s & 1] is complete; Bs[(s + 1) & 1] is free
        const int buf = s & 1;
        const f16x16 A0 = frag8(a0lo, a0hi), A1 = frag8(a1lo, a1hi);
        const bool two = 2 * s + 1 < KT;
        if (s + 1 < steps) {
            const int k0 = 2 * (s + 1);
            load16h(arow ? arow + 16 * k0 : nullptr, a0lo, a0hi);
            load16h(arow && k0 + 1 < KT ? arow + 16 * (k0 + 1) : nullptr, a1lo, a1hi);
#pragma unroll
            for (int i = 0; i < RL; ++i) {
                const int wi = i * 32 + lane;
                if (wi < TW) raw[wi] = rw[i];
            }
            if (s + 2 < steps) {
                const bool ok2 = 2 * (s + 2) + kk < KT;
                const uint32_t* tp2 = TF_DS_EXL3_TILE_OF(s + 2);
#pragma unroll
                for (int i = 0; i < RL; ++i) {
                    const int wi = i * 32 + lane;
                    rw[i] = (wi < TW && ok2) ? __builtin_nontemporal_load(gl(tp2) + wi) : 0u;
                }
            }
            lds_fence();
            uint32_t own[4];
            decode_own<K2>(raw, win, own);
            *reinterpret_cast<uint4*>(&Bs[buf ^ 1][kk][ntl][lane & 15][8 * (lane >> 4)]) = make_uint4(own[0], own[1], own[2], own[3]);
            __builtin_amdgcn_wave_barrier();
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint4 blo, bhi;
            const _Float16* c0 = &Bs[buf][0][2 * cg + j][lane & 15][0];
            blo = reinterpret_cast<const uint4*>(c0)[0];
            bhi = reinterpret_cast<const uint4*>(c0)[1];
            acc[j] = wmma(A0, frag8(blo, bhi), acc[j]);
        }
        if (two) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                uint4 blo, bhi;
                const _Float16* c1 = &Bs[buf][1][2 * cg + j][lane & 15][0];
                blo = reinterpret_cast<const uint4*>(c1)[0];
                bhi = reinterpret_cast<const uint4*>(c1)[1];
                acc[j] = wmma(A1, frag8(blo, bhi), acc[j]);
            }
        }
    }
}
#undef TF_DS_EXL3_TILE_OF

template <int LO, int HI>
__device__ __forceinline__ void grouped_mma_body(const __half* X0, const __half* X1, const int64_t* TP0, const int64_t* TP1,
                                                 const int32_t* K2_0, const int32_t* K2_1, const int32_t* uids,
                                                 const int32_t* ucount, const int32_t* members, float* Z, int K, int N, int P,
                                                 int maxm, int slots, int nexp, const int32_t* work) {
    __shared__ __attribute__((aligned(16))) _Float16 Bs[2][2][4][16][BCOL];
    __shared__ __attribute__((aligned(16))) uint32_t raw_sh[8][64];
    __shared__ int rows_sh[MMA_ROWS];
    const int zz = (int)(blockIdx.z % nexp);
    const int u = work ? work[2 * zz] : zz;
    if (u >= ucount[0]) return;
    const int mb = work ? work[2 * zz + 1] : (int)blockIdx.x;
    const int mat = (int)(blockIdx.z / nexp);
    const _Float16* X = reinterpret_cast<const _Float16*>(mat ? X1 : X0);
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int KT = K >> 4, NTILES = N >> 4;
    for (int i = threadIdx.x; i < MMA_ROWS; i += 256) {
        const int m = mb * MMA_ROWS + i;
        rows_sh[i] = member_p(m < maxm ? members[u * maxm + m] : -1, slots);
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                             // members come first, so this block is empty
    const int rg = wave & 3, cg = wave >> 2;
    const int ra = rows_sh[rg * 16 + (lane & 15)];
    const _Float16* arow = ra >= 0 ? X + (size_t)ra * K : nullptr;
    const int nt0 = blockIdx.y * (MMA_COLS / 16);
    f32x8 acc[2];
    acc[0] = acc[1] = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#define TF_DS_EXL3_MMA(K2_) mma_tiles<K2_>(T, NTILES, KT, nt0, arow, wave, lane, Bs, raw_sh[wave], acc)
    TF_DS_EXL3_K2_SWITCH(k2, LO, HI, TF_DS_EXL3_MMA)
#undef TF_DS_EXL3_MMA
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = rows_sh[rg * 16 + 2 * i + (lane >> 4)];
            if (r >= 0) Z[((size_t)mat * P + r) * N + (nt0 + 2 * cg + j) * 16 + (lane & 15)] = acc[j][i];
        }
}

// ---- exl3.experts.grouped_rows: prompt down ------------------------------------------------------------------------------
// A program per (32 member rows, 128 columns, expert place, matrix x split) from the work list (grouped_rows_kernel's
// grid: (1, N / 128, pairs x mats x splits)), four waves; wave w takes columns 32 w .. + 31 (two tiles) for both 16-row
// member tiles over the split's k tiles: a k tile's two weight tiles are decoded once (words loaded coalesced one k
// tile ahead, staged in the wave's LDS) and applied to both row tiles. Z [mats, SK, P, N].
template <int K2>
__device__ __forceinline__ void rows_tiles(const uint32_t* T, int NTILES, int kt0, int nkt, int ntw, const _Float16* arow0,
                                           const _Float16* arow1, bool live1, uint32_t* st, int lane, f32x8 (&acc)[2][2]) {
    constexpr int TW = 4 * K2, RL = (2 * TW + 31) / 32;
    const LaneWin<K2> win(lane);
    if (nkt <= 0) return;
    uint32_t pf[RL];
#define TF_DS_EXL3_LOAD_RAW(kt_)                                                                                        \
    {                                                                                                                   \
        const uint32_t* p = T + ((size_t)(kt_) * NTILES + ntw) * TW;                                                    \
        _Pragma("unroll") for (int i = 0; i < RL; ++i) {                                                               \
            const int wi = i * 32 + lane;                                                                               \
            pf[i] = wi < 2 * TW ? __builtin_nontemporal_load(gl(p) + wi) : 0u;                                              \
        }                                                                                                               \
    }
    TF_DS_EXL3_LOAD_RAW(kt0)
    uint4 a0lo, a0hi, a1lo, a1hi;
    load16h_k(arow0 ? arow0 + 16 * kt0 : nullptr, lane >> 4, a0lo, a0hi);
    load16h_k(live1 && arow1 ? arow1 + 16 * kt0 : nullptr, lane >> 4, a1lo, a1hi);
    for (int j = 0; j < nkt; ++j) {
#pragma unroll
        for (int i = 0; i < RL; ++i) {
            const int wi = i * 32 + lane;
            if (wi < 2 * TW) st[wi] = pf[i];
        }
        const f16x16 A0 = frag8(a0lo, a0hi), A1 = frag8(a1lo, a1hi);
        if (j + 1 < nkt) {
            const int kt = kt0 + j + 1;
            TF_DS_EXL3_LOAD_RAW(kt)
            load16h_k(arow0 ? arow0 + 16 * kt : nullptr, lane >> 4, a0lo, a0hi);
            load16h_k(live1 && arow1 ? arow1 + 16 * kt : nullptr, lane >> 4, a1lo, a1hi);
        }
        lds_fence();
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            uint32_t own[4];
            decode_own<K2>(st + t * TW, win, own);
            const f16x16 B = bfrag_from_own(own, lane);
            acc[0][t] = wmma(A0, B, acc[0][t]);
            if (live1) acc[1][t] = wmma(A1, B, acc[1][t]);
        }
        __builtin_amdgcn_wave_barrier();
    }
}
#undef TF_DS_EXL3_LOAD_RAW

template <int LO, int HI>
__device__ __forceinline__ void grouped_rows_body(const __half* X0, const __half* X1, const int64_t* TP0, const int64_t* TP1,
                                                  const int32_t* K2_0, const int32_t* K2_1, const int32_t* uids,
                                                  const int32_t* ucount, const int32_t* members, float* Z, int K, int N, int P,
                                                  int SK, int maxm, int slots, int nexp, const int32_t* work) {
    __shared__ __attribute__((aligned(16))) uint32_t stage[4][2 * 64];
    __shared__ int rows_sh[32];
    const int zz = (int)(blockIdx.z % nexp);
    const int u = work ? work[2 * zz] : zz;
    if (u >= ucount[0]) return;
    const int mg = work ? work[2 * zz + 1] : (int)blockIdx.x;
    const int zi = (int)(blockIdx.z / nexp);                 // (matrix, split)
    const int split = zi % SK, mat = zi / SK;
    const _Float16* X = reinterpret_cast<const _Float16*>(mat ? X1 : X0);
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int KT = K >> 4, NTILES = N >> 4;
    for (int i = threadIdx.x; i < 32; i += 128) {
        const int m = mg * 32 + i;
        rows_sh[i] = member_p(m < maxm ? members[u * maxm + m] : -1, slots);
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                             // members come first, so this group is empty
    const bool live1 = rows_sh[16] >= 0;
    const int r0 = rows_sh[lane & 15], r1 = rows_sh[16 + (lane & 15)];
    const _Float16* arow0 = r0 >= 0 ? X + (size_t)r0 * K : nullptr;
    const _Float16* arow1 = r1 >= 0 ? X + (size_t)r1 * K : nullptr;
    const int per_split = KT / SK, kt0 = split * per_split;
    const int ntw = blockIdx.y * 8 + 2 * wave;
    f32x8 acc[2][2];
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int t = 0; t < 2; ++t) acc[g][t] = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#define TF_DS_EXL3_ROWS(K2_) rows_tiles<K2_>(T, NTILES, kt0, per_split, ntw, arow0, arow1, live1, stage[wave], lane, acc)
    TF_DS_EXL3_K2_SWITCH(k2, LO, HI, TF_DS_EXL3_ROWS)
#undef TF_DS_EXL3_ROWS
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r = rows_sh[16 * g + 2 * i + (lane >> 4)];
                if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + (ntw + t) * 16 + (lane & 15)] = acc[g][t][i];
            }
}

// ---- exl3.experts.grouped_cp: decode windows ------------------------------------------------------------------------------
// grouped_cp_kernel's programs: (expert place, 128 columns, matrix x split x 16-row member tile), the last expert place
// (the shared expert) first, four waves splitting the split's k tiles (strip_tiles_1row / strip_tiles_rows over the
// expert's checkpoint layout, each wave its own contiguous range); the waves' sums added in wave order through LDS,
// rows 0-7 then 8-15.
// EPI 1 (gate / up): partials to Z [mats, SK, P, N]; the program completing an (expert, column block) runs the gate / up
// epilogue for the expert's rows there, and the expert's last such program publishes ready[place] = epoch. EPI 2 (down,
// one split): waits for its expert's ready == epoch, writes its rows' per-slot outputs (H, * svh_d) to y, and the
// program completing a (row, column block) adds the row's slots with wts in slot order into out. Counters are reset by
// their last program.
template <int LO, int HI, int EPI>
__device__ __forceinline__ void grouped_cp_body(const __half* X0, const __half* X1, const int64_t* TP0, const int64_t* TP1,
                                                const int32_t* K2_0, const int32_t* K2_1, const int32_t* uids,
                                                const int32_t* ucount, const int32_t* members, float* Z, int K, int N, int P,
                                                int SK, int maxm, int slots, const tf_ds_exl3_decode_epi& ep) {
    constexpr int W = 4, RR = 8;
    __shared__ __attribute__((aligned(16))) uint32_t stage[W][8 * 4 * HI];
    __shared__ __attribute__((aligned(16))) float red[W][RR][128];
    __shared__ int rows_sh[16];
    __shared__ int last_sh;
    const int MT = (maxm + 15) / 16;
    const int NY = gridDim.y, NZ = gridDim.z;
    const int lin = blockIdx.x + gridDim.x * (blockIdx.y + NY * blockIdx.z);
    const int ur = lin / (NY * NZ), rem = lin % (NY * NZ);
    const int nu = ucount[0];
    if (ur >= nu) return;
    const int u = nu - 1 - ur, by = rem % NY, bz = rem / NY;
    const int mtile = bz % MT;
    const int split = (bz / MT) % SK;
    const int mat = bz / MT / SK;
    const _Float16* X = reinterpret_cast<const _Float16*>(mat ? X1 : X0);
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int KT = K >> 4, NTILES = N >> 4;
    const int* pick = reinterpret_cast<const int*>(ep.pick);

    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        rows_sh[threadIdx.x] = member_p(m < maxm ? members[u * maxm + m] : -1, slots);
    }
    int cu = 0;                                             // EPI 1: the expert's member count (members are a prefix)
    if constexpr (EPI == 1) cu = __syncthreads_count(threadIdx.x < maxm && members[u * maxm + threadIdx.x] >= 0);
    else __syncthreads();
    if (rows_sh[0] < 0) return;                             // members come first, so this tile is empty
    if constexpr (EPI == 2) {
        if (ep.ready) {                                     // the expert's Xd rows (the gate / up launch ran before)
            const int epoch_v = *reinterpret_cast<const int*>(ep.epoch);
            if (threadIdx.x == 0)
                while (ld_acquire(reinterpret_cast<const int*>(ep.ready) + u) != epoch_v) __builtin_amdgcn_s_sleep(2);
            __syncthreads();
        }
    }
    // the tile's member rows (a prefix), each in the 1-row order: a row's outputs do not depend on who shares its expert;
    // the rows' staging is the wave's first 128 floats of red, free until the reduction (its barrier)
    const int nrows = __builtin_popcount((uint32_t)__ballot(lane < 16 && rows_sh[lane & 15] >= 0));
    const int ra = rows_sh[min(lane >> 1, nrows - 1)];
    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + wave * per_warp;
    const int nt0 = by * 8;
    f32x8 acc[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) acc[t] = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#define TF_DS_EXL3_CP(K2_)                                                                                              \
    if (nrows == 1) strip_tiles_1row<K2_>(T + (size_t)nt0 * (4 * K2_), (int64_t)NTILES * (4 * K2_), kt0, per_warp, X + (size_t)rows_sh[0] * K, stage[wave], lane, acc); \
    else strip_tiles_rows<K2_>(T + (size_t)nt0 * (4 * K2_), (int64_t)NTILES * (4 * K2_), kt0, per_warp, X + (size_t)ra * K, nrows, stage[wave], reinterpret_cast<uint4*>(&red[wave][0][0]), lane, acc)
    TF_DS_EXL3_K2_SWITCH(k2, LO, HI, TF_DS_EXL3_CP)
#undef TF_DS_EXL3_CP

#pragma unroll
    for (int hp = 0; hp < 2; ++hp) {
        if (hp > 0 && rows_sh[8] < 0) break;
        __syncthreads();                                    // red is reused by the second half
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[wave][2 * i + (lane >> 4)][t * 16 + (lane & 15)] = acc[t][4 * hp + i];
        __syncthreads();
        if constexpr (EPI != 2) {
            for (int idx = threadIdx.x; idx < RR * 128; idx += W * 32) {
                const int row = idx >> 7, col = idx & 127;
                const int r = rows_sh[row + RR * hp];
                if (r < 0) continue;
                float s = red[0][row][col];
#pragma unroll
                for (int w = 1; w < W; ++w) s += red[w][row][col];
                Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
            }
        } else {
            const int n = by * 128 + 4 * lane;
            const __half* svh_d = reinterpret_cast<const __half*>(ep.svh_d);
            float* y = reinterpret_cast<float*>(ep.y);
            const float* wts = reinterpret_cast<const float*>(ep.wts);
            for (int row = wave; row < RR; row += W) {
                const int p = rows_sh[row + RR * hp];
                if (p < 0) continue;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float s = red[0][row][4 * lane + j];
#pragma unroll
                    for (int w = 1; w < W; ++w) s += red[w][row][4 * lane + j];
                    v[j] = s;
                }
                fwht128(v, lane);
                float* yo = y + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) yo[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
                if (wts == nullptr) continue;
                __threadfence();
                const int r = p / slots;
                int last = 0;
                if (lane == 0) {
                    int target = 0;
                    for (int q = 0; q < slots; ++q) {
                        const int pe = pick[r * slots + q];
                        target += pe >= 0 && pe < ep.E;
                    }
                    int* c = reinterpret_cast<int*>(ep.cnt) + r * NY + by;
                    const int before = atomicAdd(c, 1);
                    last = before == target - 1;
                    if (last) *c = 0;
                }
                last = __shfl(last, 0, 32);
                if (!last) continue;
                __threadfence();
                float a4[4] = {0.f, 0.f, 0.f, 0.f};
                for (int q = 0; q < slots; ++q) {
                    const float w = wts[r * slots + q];
                    const int pq = r * slots + q;
                    const int pe = pick[pq];
                    float yv[4] = {0.f, 0.f, 0.f, 0.f};
                    if ((pe >= 0 && pe < ep.E) || ep.store_y)
#pragma unroll
                        for (int j = 0; j < 4; ++j) yv[j] = ld_cg(y + (size_t)pq * N + n + j);
#pragma unroll
                    for (int j = 0; j < 4; ++j) a4[j] = fmaf(w, yv[j], a4[j]);
                }
                const float* add = reinterpret_cast<const float*>(ep.add);
                float* out = reinterpret_cast<float*>(ep.out);
#pragma unroll
                for (int j = 0; j < 4; ++j) out[(size_t)r * N + n + j] = add ? a4[j] + add[(size_t)r * N + n + j] : a4[j];
            }
        }
    }
    if constexpr (EPI == 1) {
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            const int target = (NZ / MT) * ((cu + 15) / 16);      // mats x splits programs a live member tile
            int* c = reinterpret_cast<int*>(ep.cnt) + u * NY + by;
            const int before = atomicAdd(c, 1);
            last_sh = before == target - 1;
            if (last_sh) *c = 0;
        }
        __syncthreads();
        if (!last_sh) return;
        __threadfence();
        const int n = by * 128 + 4 * lane;
        for (int m = wave; m < cu; m += W) {
            const int p = member_p(members[u * maxm + m], slots);
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float sg = 0.f, su = 0.f;
                for (int s = 0; s < SK; ++s) {
                    sg += ld_cg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                    su += ld_cg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                }
                gv[j] = sg;
                uv[j] = su;
            }
            gateup_act(gv, uv, lane, reinterpret_cast<const __half*>(ep.svh_g), reinterpret_cast<const __half*>(ep.svh_u),
                       reinterpret_cast<const __half*>(ep.suh_d), e, N, n, ep.limit, ep.act_mode,
                       reinterpret_cast<__half*>(ep.xd) + (size_t)p * N + n);
        }
        if (ep.ready) {
            // the expert's last column block publishes its rows to down's programs (they wait on ready[place])
            __threadfence();
            __syncthreads();
            int* ready_cnt = reinterpret_cast<int*>(ep.ready_cnt);
            if (threadIdx.x == 0) last_sh = atomicAdd(ready_cnt + u, 1) == NY - 1;
            __syncthreads();
            if (threadIdx.x == 0 && last_sh) {
                ready_cnt[u] = 0;
                __threadfence();
                st_release(reinterpret_cast<int*>(ep.ready) + u, *reinterpret_cast<const int*>(ep.epoch));
            }
        }
    }
}

}  // namespace

#define TF_DS_EXL3_GROUPED_ARGS                                                                                         \
    const __half *X0, const __half *X1, const int64_t *TP0, const int64_t *TP1, const int32_t *K2_0, const int32_t *K2_1, \
        const int32_t *uids, const int32_t *ucount, const int32_t *members, float *Z, int32_t K, int32_t N, int32_t P

#define TF_DS_EXL3_GROUPED(LO_, HI_)                                                                                    \
    extern "C" __global__ void __launch_bounds__(256) tf_ds_exl3_experts_grouped_mma_lo##LO_##_hi##HI_(                \
        TF_DS_EXL3_GROUPED_ARGS, int32_t SK, int32_t WK, int32_t maxm, int32_t slots, int32_t nexp, const int32_t* work) { \
        (void)SK;                                                                                                       \
        (void)WK;                                                                                                       \
        grouped_mma_body<LO_, HI_>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, maxm, slots, nexp, work); \
    }                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(128) tf_ds_exl3_experts_grouped_rows_lo##LO_##_hi##HI_(               \
        TF_DS_EXL3_GROUPED_ARGS, int32_t SK, int32_t maxm, int32_t slots, int32_t nexp, const int32_t* work) {          \
        grouped_rows_body<LO_, HI_>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots, nexp, work); \
    }                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(128) tf_ds_exl3_experts_grouped_cp_lo##LO_##_hi##HI_##_epi1(          \
        TF_DS_EXL3_GROUPED_ARGS, int32_t SK, int32_t maxm, int32_t slots, tf_ds_exl3_decode_epi ep) {                   \
        grouped_cp_body<LO_, HI_, 1>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots, ep); \
    }                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(128) tf_ds_exl3_experts_grouped_cp_lo##LO_##_hi##HI_##_epi2(          \
        TF_DS_EXL3_GROUPED_ARGS, int32_t SK, int32_t maxm, int32_t slots, tf_ds_exl3_decode_epi ep) {                   \
        grouped_cp_body<LO_, HI_, 2>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots, ep); \
    }
TF_DS_EXL3_GROUPED(8, 8)
TF_DS_EXL3_GROUPED(2, 10)
TF_DS_EXL3_GROUPED(2, 16)   // the MTP blocks' sets: routed K2 8 with the K2 12 shared expert
#undef TF_DS_EXL3_GROUPED
#undef TF_DS_EXL3_GROUPED_ARGS

// ==== exl3.l2_prefetch: the next kernels' weights into gfx1151's memory-side cache (ops.zig l2Prefetch) ===========
// The AMD side of the served build's paced L2 prefetch (round.zig prefetch: forked onto the round's prefetch stream after
// a launch, joined at the stretch's end): while the main stream runs gathers and small kernels (DRAM idle ~9 ms of a
// 31 ms drafts-off token), the ranges are read with plain loads, which allocate in the 32 MB MALL; the weight kernels'
// nontemporal loads then hit it (16 MB in 22 us after a touch vs 78 us cold; tools/dsv41-hip/exl3/KDEC.md 3). Writes
// nothing. The CUDA kernel's parameters and meaning: after delay_ns, at most 1 MiB is started every wave_ns (0:
// unpaced; ops.zig's 4,559 ns is 230 GB/s); each range from its first byte, cut to whole 16-byte units, starting on a
// chunk of its own. The grid's blocks take the 16 KiB chunks in order (block b: chunks b, b + G, ...), each thread four
// coalesced 16-byte loads a chunk; the values meet an empty asm so the loads stay. gfx1151's constant clock (s_memrealtime,
// wall_clock64) runs at 100 MHz (hipDeviceAttributeWallClockRate 99,810 kHz): 10 ns a tick.
namespace {
constexpr int64_t kPfChunk = 16 << 10;
constexpr int64_t kPfNsPerTick = 10;
}  // namespace

extern "C" __global__ void __launch_bounds__(256) tf_ds_exl3_l2_prefetch(tf_ds_prefetch a, int64_t delay_ns, int64_t wave_ns) {
    const int64_t t0 = (int64_t)wall_clock64();
    int64_t end[16];
    int64_t total = 0;
    const int nr = a.n < 16 ? a.n : 16;
    for (int r = 0; r < nr; ++r) {
        total += ((int64_t)(a.bytes[r] / 16 * 16) + kPfChunk - 1) / kPfChunk;
        end[r] = total;
    }
    int r = 0;
    for (int64_t c = blockIdx.x; c < total; c += gridDim.x) {
        // pacing: chunk c starts once delay_ns + (c * 16 KiB / 1 MiB) * wave_ns has passed
        const int64_t due_ns = delay_ns + (wave_ns > 0 ? (c * kPfChunk >> 20) * wave_ns : 0);
        if (due_ns > 0)
            while (((int64_t)wall_clock64() - t0) * kPfNsPerTick < due_ns) __builtin_amdgcn_s_sleep(2);
        while (c >= end[r]) ++r;
        const int64_t first = r ? end[r - 1] : 0;
        const uint64_t base = a.ptr[r] + (uint64_t)(c - first) * kPfChunk;
        const uint64_t lim = a.ptr[r] + a.bytes[r] / 16 * 16;
        uint32_t acc = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint64_t at = base + (uint64_t)i * 4096 + (uint64_t)threadIdx.x * 16;
            if (at < lim) {
                const uint4 v = *gl(reinterpret_cast<const uint4*>(at));
                acc ^= v.x ^ v.w;
            }
        }
        asm volatile("" ::"v"(acc));
    }
}
