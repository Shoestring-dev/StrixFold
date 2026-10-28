// ABI echo kernels (tf-dsv41-hip-abi-test): each writes back exactly the bytes or values it received, so the host can
// compare every field it packed through gpu.Kernel against what the device read. Test kit only.
#include <hip/hip_runtime.h>
#include "abi.h"

// The by-value struct's bytes, as the kernel sees them.
extern "C" __global__ void __launch_bounds__(64) tf_ds_abi_echo_mixed(const tf_ds_abi_mixed a, uint8_t* out) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&a);
    for (uint32_t i = threadIdx.x; i < sizeof(a); i += blockDim.x) out[i] = p[i];
}

extern "C" __global__ void __launch_bounds__(256) tf_ds_abi_echo_big(const tf_ds_abi_big a, uint8_t* out) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&a);
    for (uint32_t i = threadIdx.x; i < sizeof(a); i += blockDim.x) out[i] = p[i];
}

// w-exl3's by-value structs (abi.h): glinear's and rot_many's whole arguments, and grouped_cp's list (ten addresses, six
// ints, the 144-byte epilogue at offset 104) read back from the kernarg segment as the kernel received it.
extern "C" __global__ void __launch_bounds__(256) tf_ds_abi_echo_exl3_gargs(const tf_ds_exl3_gargs a, uint8_t* out) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&a);
    for (uint32_t i = threadIdx.x; i < sizeof(a); i += blockDim.x) out[i] = p[i];
}

extern "C" __global__ void __launch_bounds__(256) tf_ds_abi_echo_exl3_rargs(const tf_ds_exl3_rargs a, uint8_t* out) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&a);
    for (uint32_t i = threadIdx.x; i < sizeof(a); i += blockDim.x) out[i] = p[i];
}

extern "C" __global__ void __launch_bounds__(64) tf_ds_abi_echo_exl3_cp(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,
                                                                        uint64_t a5, uint64_t a6, uint64_t a7, uint64_t a8, uint64_t a9,
                                                                        int32_t k, int32_t n, int32_t p, int32_t splits, int32_t maxm,
                                                                        int32_t slots, const tf_ds_exl3_decode_epi epi, uint8_t* out) {
    const uint8_t* ka = static_cast<const uint8_t*>(__builtin_amdgcn_kernarg_segment_ptr());
    for (uint32_t i = threadIdx.x; i < 104 + sizeof(epi); i += blockDim.x) out[i] = ka[i];
}

// Scalars of every width in an order that needs padding between them; each lands in its own 8-byte slot of out.
extern "C" __global__ void __launch_bounds__(32) tf_ds_abi_echo_params(uint8_t a, uint64_t b, uint16_t c, float d, int8_t e,
                                                                       double f, int32_t g, uint64_t out) {
    if (threadIdx.x != 0) return;
    uint64_t* o = reinterpret_cast<uint64_t*>(out);
    o[0] = a;
    o[1] = b;
    o[2] = c;
    o[3] = __float_as_uint(d);
    o[4] = (uint64_t)(int64_t)e;
    o[5] = __double_as_longlong(f);
    o[6] = (uint64_t)(int64_t)g;
}

// Stand-ins for two AOT Triton variants of one function `_echo` (runtime parameters X, n, s, OUT, then Triton's two
// hidden scratch pointers): each writes its tag and what it received, so the test sees which variant the set chose.
template <uint64_t TAG>
__device__ void tri_echo(const void* X, int32_t n, float s, uint64_t* OUT, void* gs, void* ps) {
    if (threadIdx.x != 0) return;
    OUT[0] = TAG;
    OUT[1] = (uint64_t)X;
    OUT[2] = (uint64_t)(int64_t)n;
    OUT[3] = __float_as_uint(s);
    OUT[4] = (uint64_t)gs;
    OUT[5] = (uint64_t)ps;
    OUT[6] = blockDim.x;
}
extern "C" __global__ void tf_ds_abi_tri_echo_a(const void* X, int32_t n, float s, uint64_t* OUT, void* gs, void* ps) {
    tri_echo<0xA>(X, n, s, OUT, gs, ps);
}
extern "C" __global__ void tf_ds_abi_tri_echo_b(const void* X, int32_t n, float s, uint64_t* OUT, void* gs, void* ps) {
    tri_echo<0xB>(X, n, s, OUT, gs, ps);
}
