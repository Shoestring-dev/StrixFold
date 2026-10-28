// gfx11 wave32 v_wmma_f32_16x16x16_bf16 operand layouts, probed: A row-major from a[16][16], B from b[16][16] (K x N),
// D = A B + C written out per lane as the layout assumed by aux attention (lane l: column l % 16, rows 2i + l / 16).
#include <hip/hip_runtime.h>
#include <stdint.h>
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
extern "C" __global__ __launch_bounds__(32) void tf_ds_aux_wmma_probe(const float* a, const float* b, float* d) {
    const int l = threadIdx.x;
    bf16x16 va, vb;
    for (int k = 0; k < 16; ++k) {
        va[k] = (__bf16)a[(l % 16) * 16 + k];        // lane l: row l % 16 of A, its 16 k
        vb[k] = (__bf16)b[k * 16 + (l % 16)];        // lane l: column l % 16 of B, its 16 k
    }
    f32x8 c = {0, 0, 0, 0, 0, 0, 0, 0};
    c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va, vb, c);
    for (int i = 0; i < 8; ++i) d[(2 * i + l / 16) * 16 + (l % 16)] = c[i];
}
