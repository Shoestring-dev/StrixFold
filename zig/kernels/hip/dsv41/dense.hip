/* p-par's dense projections for the dsv41 family on HIP (registered in registry.json as group "dense"; dense.zig
 * launches them, hip_load.zig binds them). Some fine-tuned packs store attn.wo_b and ffn.shared_experts.w2
 * of layers 10-36 as BF16 weights [N, K] instead of EXL3 parts; each rank holds the K rows of its split (plan.zig).
 *
 * tf_ds_dense_gemv: a decode / verify window's rows through one such weight (prompt chunks take hipBLAS GemmEx).
 * Row invariance (k-dec's contract for the decode paths: a row's outputs are the bits it gets when it runs alone):
 * every row's sum is the same arithmetic whatever the window's rows, its place in the window or the other rows'
 * values. Lane l of the wave that owns output column n keeps one fp32 partial per row: over the K tiles in order,
 * the tile's elements l * E .. l * E + E - 1 in order, each w * x (both bf16, exact in fp32) folded in with one fma
 * (the kit's -ffp-contract=off leaves fmaf the only fused op); the tile width (256, or 128 when K is no multiple of
 * 256, E = width / 32) depends on K alone. The 32 partials are then summed by xor shuffles 16, 8, 4, 2, 1, and lane 0
 * stores y = sum, or y = y + sum (one fp32 add: the shared expert added after the routed experts' slots, as the
 * served sum adds its weight-1 slot last). Rows run in passes of 8; a pass's composition changes no row's operations.
 *
 * tf_ds_dense_swiglu: the shared expert's activation between its EXL3 gate / up (fp32) and its dense down: the
 * reference's Expert.forward (DeepSeek inference/model.py): up clamped to [-limit, limit], gate clamped from above,
 * silu(gate) * up in fp32 (silu = g / (1 + exp(-g))), then .to(bf16) (round to nearest even) for w2. A NaN stays a
 * NaN through the clamps, as torch.clamp keeps it.
 *
 * tf_ds_dense_edit: a rank-1 edit of a projection's fp32 output rows (edit.zig: an abliterated pack's changed wo_b and
 * shared w2 expressed on the EXL3 pack's own weights), out = g * (y - alpha * r * (r . y)), or out + that (the shared
 * expert's down, added after the routed slots). Row invariance: thread t folds elements t, t + 256, ... of r * y in
 * order (one fma each), the 32 lanes' partials are summed by xor shuffles 16, 8, 4, 2, 1 and the 8 waves' in wave
 * order, the same by every thread, so a row's arithmetic depends on n alone; then s = alpha * dot and each element
 * g * fmaf(-s, r, y). */
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include "abi.h"

#define TF_DS_DENSE_WAVES 8 // a block's waves
#define TF_DS_DENSE_WCOLS 4 // output columns a wave
#define TF_DS_DENSE_PASS 8 // rows a pass

template <int E>
static __device__ inline void tf_ds_dense_load(const uint16_t* p, float (&out)[8]) {
    if constexpr (E == 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(p);
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            out[2 * j] = __uint_as_float(w[j] << 16);
            out[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
        }
    } else {
        const uint2 v = *reinterpret_cast<const uint2*>(p);
        const uint32_t w[2] = {v.x, v.y};
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            out[2 * j] = __uint_as_float(w[j] << 16);
            out[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
        }
    }
}

// A wave's 4 columns over a pass of up to 8 rows: every K tile's weights for the 4 columns are loaded first (16 or 8
// bytes a lane each: the next tile's while this one's are used), then each row's x (from L1 / L2: the window's rows
// are read by every wave) folded into its 4 partials. No LDS and no block barrier: the waves run apart.
template <int E>
static __device__ void tf_ds_dense_gemv_body(const uint16_t* w, const uint16_t* x, float* y, int32_t k, int32_t n,
                                             int32_t rows, int64_t ldx, int64_t ldy, int32_t accumulate) {
    constexpr int tile = 32 * E;
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int c0 = (blockIdx.x * TF_DS_DENSE_WAVES + wave) * TF_DS_DENSE_WCOLS;
    if (c0 >= n) return;
    const uint16_t* wc[TF_DS_DENSE_WCOLS];
#pragma unroll
    for (int c = 0; c < TF_DS_DENSE_WCOLS; ++c) wc[c] = w + int64_t(min(c0 + c, n - 1)) * k + lane * E;
    for (int r0 = 0; r0 < rows; r0 += TF_DS_DENSE_PASS) {
        const int rr = min(TF_DS_DENSE_PASS, rows - r0);
        float acc[TF_DS_DENSE_WCOLS][TF_DS_DENSE_PASS];
#pragma unroll
        for (int c = 0; c < TF_DS_DENSE_WCOLS; ++c)
#pragma unroll
            for (int r = 0; r < TF_DS_DENSE_PASS; ++r) acc[c][r] = 0.0f;
        float wv[TF_DS_DENSE_WCOLS][8];
#pragma unroll
        for (int c = 0; c < TF_DS_DENSE_WCOLS; ++c) tf_ds_dense_load<E>(wc[c], wv[c]);
        for (int t = 0; t < k; t += tile) {
            float wn[TF_DS_DENSE_WCOLS][8];
            if (t + tile < k) {
#pragma unroll
                for (int c = 0; c < TF_DS_DENSE_WCOLS; ++c) tf_ds_dense_load<E>(wc[c] + t + tile, wn[c]);
            }
#pragma unroll
            for (int r = 0; r < TF_DS_DENSE_PASS; ++r) {
                if (r >= rr) break;
                float xv[8];
                tf_ds_dense_load<E>(x + int64_t(r0 + r) * ldx + t + lane * E, xv);
#pragma unroll
                for (int c = 0; c < TF_DS_DENSE_WCOLS; ++c) {
                    float a = acc[c][r];
#pragma unroll
                    for (int j = 0; j < E; ++j) a = fmaf(wv[c][j], xv[j], a);
                    acc[c][r] = a;
                }
            }
            if (t + tile < k) {
#pragma unroll
                for (int c = 0; c < TF_DS_DENSE_WCOLS; ++c)
#pragma unroll
                    for (int j = 0; j < 8; ++j) wv[c][j] = wn[c][j];
            }
        }
#pragma unroll
        for (int c = 0; c < TF_DS_DENSE_WCOLS; ++c) {
            const int col = c0 + c;
#pragma unroll
            for (int r = 0; r < TF_DS_DENSE_PASS; ++r) {
                if (r >= rr) break;
                float v = acc[c][r];
                v += __shfl_xor(v, 16);
                v += __shfl_xor(v, 8);
                v += __shfl_xor(v, 4);
                v += __shfl_xor(v, 2);
                v += __shfl_xor(v, 1);
                if (lane == 0 && col < n) {
                    float* out = y + int64_t(r0 + r) * ldy + col;
                    *out = accumulate ? *out + v : v;
                }
            }
        }
    }
}

// y [rows, n] fp32 (rows ldy apart) = (y +) x [rows, k] bf16 (rows ldx apart) . w^T, w bf16 [n, k]; a block of 256
// threads owns 32 output columns (grid n / 32, rounded up). k a multiple of 128; w and x's rows 16-byte aligned.
extern "C" __global__ void __launch_bounds__(256) tf_ds_dense_gemv(const uint16_t* w, const uint16_t* x, float* y, int32_t k,
                                                                   int32_t n, int32_t rows, int64_t ldx, int64_t ldy,
                                                                   int32_t accumulate) {
    if (k % 256 == 0)
        tf_ds_dense_gemv_body<8>(w, x, y, k, n, rows, ldx, ldy, accumulate);
    else
        tf_ds_dense_gemv_body<4>(w, x, y, k, n, rows, ldx, ldy, accumulate);
}

// h [rows, width] bf16 (rows ldh apart) = silu(min(g, limit)) * clamp(u, -limit, limit) of g, u fp32 [rows, width]
// (rows ld apart); limit <= 0: no clamps.
extern "C" __global__ void __launch_bounds__(256) tf_ds_dense_swiglu(const float* g, const float* u, int64_t ld, __hip_bfloat16* h,
                                                                     int64_t ldh, int32_t rows, int32_t width, float limit) {
    const int64_t total = int64_t(rows) * width;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t r = i / width, j = i % width;
        float gv = g[r * ld + j];
        float uv = u[r * ld + j];
        if (limit > 0.0f) {
            uv = uv < -limit ? -limit : (uv > limit ? limit : uv);
            gv = gv > limit ? limit : gv;
        }
        const float s = gv / (1.0f + expf(-gv));
        h[r * ldh + j] = __float2bfloat16(s * uv);
    }
}

// out [rows, n] fp32 (rows ldo apart) = (out +) g * (y - alpha * r * (r . y)) of y [rows, n] fp32 (rows ldy apart),
// r and g fp32 [n]; a block of 256 threads a row (grid rows). out may be y when not accumulating: each thread writes
// only the elements it reads.
extern "C" __global__ void __launch_bounds__(256) tf_ds_dense_edit(const float* r, const float* g, float alpha,
                                                                   const float* y, int64_t ldy, float* out, int64_t ldo,
                                                                   int32_t n, int32_t accumulate) {
    __shared__ float part[8];
    const float* yr = y + int64_t(blockIdx.x) * ldy;
    float* orow = out + int64_t(blockIdx.x) * ldo;
    float acc = 0.0f;
    for (int j = threadIdx.x; j < n; j += 256) acc = fmaf(r[j], yr[j], acc);
    acc += __shfl_xor(acc, 16);
    acc += __shfl_xor(acc, 8);
    acc += __shfl_xor(acc, 4);
    acc += __shfl_xor(acc, 2);
    acc += __shfl_xor(acc, 1);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    float dot = part[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) dot += part[w];
    const float s = alpha * dot;
    for (int j = threadIdx.x; j < n; j += 256) {
        const float v = g[j] * fmaf(-s, r[j], yr[j]);
        orow[j] = accumulate ? orow[j] + v : v;
    }
}
