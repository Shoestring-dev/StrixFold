/* w-arch's glue kernels for the dsv41 family on HIP (ADAPTER.md section 6: group "glue", abi "params", fixed-width
 * parameters, registered in registry.json; ops.zig launches them, hip_load.zig binds them). Ported from TensorFold's
 * CUDA sources at strix-port-base (Apache-2.0):
 *   zig/kernels/cuda/torch_ops/pointwise.cu (sha256 3d3118159265538679968e56a8c4b5d61b01c07487530e95e4f93ba8662f98c8):
 *     bf16_to_f32
 *   zig/kernels/cuda/torch_ops/movement.cu (sha256 aa47a7127e6853aa4e28068de38690fbec3afbd965c591625c5bf6ed386012f2):
 *     gather_rows, strided_copy
 *   zig/kernels/cuda/dsv41_ops.cu (sha256 728ef4b0784143736945d8ecb44397f6c25af5db3445ab523ee99c0b8946b0a6):
 *     f16_to_f32, add2_bf16, scatter_rows, copy_rows16, argmax_rows, clock
 * Kernel bodies are the CUDA ones; changes: names (tf_<op>_kernel / tf_ds_<op>_kernel -> tf_ds_glue_<op>), parameter
 * types spelled with fixed widths (the same sizes, so ops.zig packs them unchanged), __launch_bounds__ at the block
 * ops.zig launches, bf16 as HIP's __hip_bfloat16 (as aux.hip), argmax_rows' __shfl_down_sync as the wave32 __shfl_down,
 * and the clock (below). The kit's flags are -ffp-contract=off -fdenormal-fp-math=ieee (the CUDA build's --fmad=false
 * --ftz=false); add2_bf16's one fp32 add is a plain IEEE add.
 *
 * bf16 NaN: `.to(bf16)` here and in aux.hip is clang's float -> __bf16 conversion, which keeps a NaN's sign and high
 * payload bits and sets the quiet bit; CUDA's cvt.rn.bf16.f32 gives a canonical NaN instead. Every finite and infinite
 * value converts identically (round to nearest even); a NaN stays a NaN (glue_test checks both). */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include "abi.h"

typedef __hip_bfloat16 tf_ds_bf16;

// `.float()` of bf16 values (exact): the high half of the float's bits, as __bfloat162float on CUDA (a NaN's payload
// is kept bit for bit).
extern "C" __global__ void __launch_bounds__(256) tf_ds_glue_bf16_to_f32(const uint16_t* input, float* output, uint64_t count) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += uint64_t(gridDim.x) * blockDim.x)
        output[i] = __uint_as_float(uint32_t(input[i]) << 16);
}

// `.float()` of fp16 values (exact): the MoE gate's weights before torch's fp32 GEMM (model.py moe, prompt rows).
extern "C" __global__ void __launch_bounds__(256) tf_ds_glue_f16_to_f32(const __half* input, float* output, uint64_t count) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += uint64_t(gridDim.x) * blockDim.x)
        output[i] = __half2float(input[i]);
}

// Comm.sum of two ranks' fp32 partials, then `.to(bf16)`: acc = g0 + g1 (one fp32 add, rank order), rounded to nearest
// even into bf16 (model.py engram_apply: comm.sum(mm(engram_wkv, e, F32)).to(BF16)).
extern "C" __global__ void __launch_bounds__(256) tf_ds_glue_add2_bf16(const float* g0, const float* g1, tf_ds_bf16* output,
                                                                       uint64_t count) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += uint64_t(gridDim.x) * blockDim.x)
        output[i] = __float2bfloat16(g0[i] + g1[i]);
}

// t[idx] = rows of whole rows (index_put_ with distinct indices: the window ring's and the positional store's slots):
// dst row idx[r] = src row r, `bytes` bytes a row (16-byte words when the rows are aligned so), a row a block.
extern "C" __global__ void __launch_bounds__(256) tf_ds_glue_scatter_rows(const uint8_t* src, int64_t src_ld, const int64_t* idx,
                                                                          uint8_t* dst, int64_t dst_ld, int64_t bytes, int32_t rows) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const uint8_t* s = src + (int64_t)r * src_ld;
    uint8_t* d = dst + idx[r] * dst_ld;
    if ((((uintptr_t)s | (uintptr_t)d | (uintptr_t)bytes) & 15) == 0) {
        const uint4* s4 = (const uint4*)s;
        uint4* d4 = (uint4*)d;
        for (int64_t j = threadIdx.x; j < bytes / 16; j += blockDim.x) d4[j] = s4[j];
    } else {
        for (int64_t j = threadIdx.x; j < bytes; j += blockDim.x) d[j] = s[j];
    }
}

// dst row r = src row indices[r] (`t[idx]`), a row a block; an index outside [0, source_rows) sets *invalid and leaves
// that row as it was.
extern "C" __global__ void __launch_bounds__(256) tf_ds_glue_gather_rows(const uint8_t* source, uint8_t* destination,
                                                                         const int64_t* indices, uint64_t rows, uint64_t source_rows,
                                                                         uint64_t row_bytes, uint32_t* invalid) {
    const uint64_t row = blockIdx.x;
    if (row >= rows) return;
    const int64_t pick = indices[row];
    if (pick < 0 || uint64_t(pick) >= source_rows) {
        if (threadIdx.x == 0) atomicExch(invalid, 1u);
        return;
    }
    for (uint64_t i = threadIdx.x; i < row_bytes; i += blockDim.x)
        destination[row * row_bytes + i] = source[uint64_t(pick) * row_bytes + i];
}

// `.contiguous()` of an [outer, middle, inner_bytes] byte view with the given outer and middle strides, a block an
// outer group and a byte a thread.
extern "C" __global__ void __launch_bounds__(256) tf_ds_glue_strided_copy(const uint8_t* source, uint8_t* destination,
                                                                          uint64_t outer, uint64_t middle, uint64_t inner_bytes,
                                                                          uint64_t source_outer, uint64_t source_middle,
                                                                          uint64_t destination_outer, uint64_t destination_middle) {
    const uint64_t group = blockIdx.x;
    const uint64_t total = middle * inner_bytes;
    for (uint64_t i = threadIdx.x; group < outer && i < total; i += blockDim.x) {
        const uint64_t row = i / inner_bytes;
        const uint64_t byte = i % inner_bytes;
        destination[group * destination_outer + row * destination_middle + byte] =
            source[group * source_outer + row * source_middle + byte];
    }
}

// Rows of n16 16-byte words: dst row r = src row r (row strides in words), a word a thread over grid
// (ceil(n16 / blockDim), rows). Every pointer, stride and row length a multiple of 16 bytes (ops.copyRows checks).
extern "C" __global__ void __launch_bounds__(256) tf_ds_glue_copy_rows16(const uint4* src, uint64_t src_ld, uint4* dst,
                                                                         uint64_t dst_ld, uint64_t n16) {
    const uint64_t r = blockIdx.y;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n16) dst[r * dst_ld + i] = src[r * src_ld + i];
}

// A greedy row's token as sampling.argmax takes it on the host (best = 0; best = i where row[i] > row[best]): the
// first index of the largest value, a NaN never larger, and 0 when the row's first value is NaN. logits [rows, n] fp32
// (row stride ld), out [rows] u32; a row a block (1024 threads, 32 waves of 32).
extern "C" __global__ void __launch_bounds__(1024) tf_ds_glue_argmax_rows(const float* logits, uint64_t ld, int32_t n,
                                                                          uint32_t* out) {
    const float* row = logits + (uint64_t)blockIdx.x * ld;
    const float ninf = -__int_as_float(0x7f800000);
    float bv = ninf;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i];
        if (v == v && (v > bv || (v == bv && i < bi))) {
            bv = v;
            bi = i;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down(bv, off, 32);
        const int oi = __shfl_down(bi, off, 32);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp != 0) return;
    const int nw = (blockDim.x + 31) >> 5;
    bv = lane < nw ? sv[lane] : ninf;
    bi = lane < nw ? si[lane] : 0x7fffffff;
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down(bv, off, 32);
        const int oi = __shfl_down(bi, off, 32);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) out[blockIdx.x] = row[0] != row[0] ? 0u : (uint32_t)bi;
}

// A profile's GPU clock (ns) when the stream reaches it, appended with its tag: buf[0] the count (reset first when
// `reset`), then (clock, tag) pairs, at most `cap` (round phase timing; in a graph like any kernel). CUDA reads
// %globaltimer (ns); here the device's constant-rate wall clock (wall_clock64, hipDeviceAttributeWallClockRate:
// 99.81 MHz on gfx1151) times ns_q32, the nanoseconds a tick in 32.32 fixed point (ops.zig passes it from the caps).
extern "C" __global__ void __launch_bounds__(32) tf_ds_glue_clock(uint64_t* buf, uint32_t tag, uint32_t reset, uint32_t cap,
                                                                  uint64_t ns_q32) {
    if (threadIdx.x == 0) {
        const uint64_t ticks = wall_clock64();
        const uint64_t t = (__umul64hi(ticks, ns_q32) << 32) | ((ticks * ns_q32) >> 32);
        const uint64_t i = reset ? 0 : buf[0];
        if (i < cap) {
            buf[1 + 2 * i] = t;
            buf[2 + 2 * i] = tag;
            buf[0] = i + 1;
        }
    }
}
