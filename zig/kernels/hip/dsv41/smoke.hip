// Kernels of the HIP runtime smoke test (tf-dsv41-hip-smoke): each one makes one runtime behaviour observable.
// Not part of the serving kit.
#include <hip/hip_runtime.h>
#include <stdint.h>

// out[i] = i + bias + (tag >> 32): u32, u32 and u64 parameters packed by Args.
extern "C" __global__ void __launch_bounds__(256) tf_smoke_probe(uint32_t* out, uint32_t n, uint32_t bias, uint64_t tag) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i + bias + (uint32_t)(tag >> 32);
}

// One thread adds to a counter: a chain of these is a chain of dependent launches (graph replay counts).
extern "C" __global__ void tf_smoke_step(uint64_t* counter, uint64_t add) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *counter += add;
}

// Spins until `ticks` of the constant wall clock have passed, then writes `value`: queued work a host query can see.
extern "C" __global__ void tf_smoke_wait_write(uint32_t* out, uint32_t value, uint64_t ticks) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint64_t t0 = wall_clock64();
        while (wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(1);
        out[0] = value;
    }
}

// The wavefront width the code object runs at.
extern "C" __global__ void tf_smoke_wave(int32_t* out) {
    if (threadIdx.x == 0) out[0] = warpSize;
}

// out[i] = in[i] ^ 0x5a5a5a5a: reads host-mapped memory and writes device memory (or the reverse).
extern "C" __global__ void __launch_bounds__(256) tf_smoke_xor(const uint32_t* in, uint32_t* out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] ^ 0x5a5a5a5au;
}

__device__ int32_t tf_smoke_table[4];

// out[t] = table[t] * 2 for the module global a host write filled.
extern "C" __global__ void tf_smoke_read_table(int32_t* out) {
    if (threadIdx.x < 4) out[threadIdx.x] = tf_smoke_table[threadIdx.x] * 2;
}

// Toolchain probe (w-exl3's finding on clang 23 / HIP 7.15, gfx1151): a builtin bit cast of an f16x2 vector's lane 1 to
// uint16_t silently returns lane 0. out[0..1]: bit_cast of lanes 1 and 0; out[2..3]: the same lanes through memcpy, the
// idiom the kit uses. Inputs come from memory so nothing is folded at compile time.
typedef _Float16 tf_f16x2 __attribute__((ext_vector_type(2)));
extern "C" __global__ void tf_smoke_f16x2_lanes(const uint16_t* in, uint32_t* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    tf_f16x2 v;
    __builtin_memcpy(&v, in, sizeof(v));
    out[0] = __builtin_bit_cast(uint16_t, v[1]); // bitcast-ok: the probe of the bug itself
    out[1] = __builtin_bit_cast(uint16_t, v[0]); // bitcast-ok: the probe of the bug itself
    uint16_t lanes[2];
    __builtin_memcpy(lanes, &v, sizeof(lanes));
    out[2] = lanes[1];
    out[3] = lanes[0];
}
