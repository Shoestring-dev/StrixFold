// w-tri: memory bandwidth probe (streaming write / read / copy / read-read-write) on gfx1151
#include <hip/hip_runtime.h>
#include <stdint.h>
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
extern "C" __global__ __launch_bounds__(256) void bw_write16(u32x4* out, int64_t n16) {
    for (int64_t i = blockIdx.x * 256 + threadIdx.x; i < n16; i += (int64_t)gridDim.x * 256) out[i] = u32x4{(uint32_t)i, 1, 2, 3};
}
extern "C" __global__ __launch_bounds__(256) void bw_write8(u32x2* out, int64_t n8) {
    for (int64_t i = blockIdx.x * 256 + threadIdx.x; i < n8; i += (int64_t)gridDim.x * 256) out[i] = u32x2{(uint32_t)i, 1};
}
extern "C" __global__ __launch_bounds__(256) void bw_read16(const u32x4* in, uint32_t* sink, int64_t n16) {
    uint32_t acc = 0;
    for (int64_t i = blockIdx.x * 256 + threadIdx.x; i < n16; i += (int64_t)gridDim.x * 256) { const u32x4 v = in[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x12345678u) sink[0] = acc;
}
extern "C" __global__ __launch_bounds__(256) void bw_copy16(const u32x4* in, u32x4* out, int64_t n16) {
    for (int64_t i = blockIdx.x * 256 + threadIdx.x; i < n16; i += (int64_t)gridDim.x * 256) out[i] = in[i];
}
// two reads (a: n16, b: 2 n16) and one write (n16): the mHC posted-mix traffic shape (x + g -> xo)
extern "C" __global__ __launch_bounds__(256) void bw_rrw16(const u32x4* a, const u32x4* b, u32x4* out, int64_t n16) {
    for (int64_t i = blockIdx.x * 256 + threadIdx.x; i < n16; i += (int64_t)gridDim.x * 256) {
        const u32x4 x = a[i], y0 = b[2 * i], y1 = b[2 * i + 1];
        out[i] = x ^ y0 ^ y1;
    }
}
