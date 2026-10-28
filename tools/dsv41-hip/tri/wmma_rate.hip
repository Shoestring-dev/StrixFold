// w-tri: WMMA throughput probe (v_wmma_f32_16x16x16_bf16, wave32): each wave runs iters x 8 independent WMMAs
#include <hip/hip_runtime.h>
#include <stdint.h>
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
extern "C" __global__ __launch_bounds__(256) void wmma_rate(float* out, int iters) {
    bf16x16 a, b;
    for (int i = 0; i < 16; ++i) {
        a[i] = (__bf16)(float)((threadIdx.x + i) & 3);
        b[i] = (__bf16)(float)((threadIdx.x * 3 + i) & 3);
    }
    f32x8 c[8];
    for (int j = 0; j < 8; ++j) c[j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 8; ++j) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c[j]);
    }
    float s = 0.f;
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i) s += c[j][i];
    if (s == 1234.5f) out[0] = s;
}
// the same with one dependent chain (latency)
extern "C" __global__ __launch_bounds__(256) void wmma_chain(float* out, int iters) {
    bf16x16 a, b;
    for (int i = 0; i < 16; ++i) {
        a[i] = (__bf16)(float)((threadIdx.x + i) & 3);
        b[i] = (__bf16)(float)((threadIdx.x * 3 + i) & 3);
    }
    f32x8 c = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    for (int it = 0; it < iters * 8; ++it) c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    float s = 0.f;
    for (int i = 0; i < 8; ++i) s += c[i];
    if (s == 1234.5f) out[0] = s;
}
