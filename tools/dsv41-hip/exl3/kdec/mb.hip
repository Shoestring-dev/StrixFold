// k-dec microbenchmark: issue cost of the decode's VALU instructions on gfx1151 (8 independent chains a lane, every
// op by inline asm so exactly that instruction issues). Time per op relative to v_add_nc_u32 at the same clock.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s line %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)

#define CHAINS(OPSTR)                                                                                                   \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) asm volatile(OPSTR : "+v"(a[j]) : "v"(c), "v"(d));

template <int OP>
__global__ void __launch_bounds__(256) mb(uint32_t* out, uint32_t seed, int n) {
    uint32_t a[8];
    const uint32_t c = seed * 3u + 0x83DCD12Du, d = seed ^ 0x64006400u;
    for (int j = 0; j < 8; ++j) a[j] = seed + threadIdx.x * 8 + j + blockIdx.x;
    for (int i = 0; i < n; ++i) {
        if constexpr (OP == 0) { CHAINS("v_add_nc_u32 %0, %0, %1") }
        if constexpr (OP == 1) { CHAINS("v_mul_lo_u32 %0, %0, %1") }
        if constexpr (OP == 2) { CHAINS("v_mul_u32_u24 %0, %0, %1") }
        if constexpr (OP == 3) { CHAINS("v_mad_u32_u16 %0, %0, %1, %2") }
        if constexpr (OP == 4) { CHAINS("v_sad_u8 %0, %0, 0, %1") }
        if constexpr (OP == 5) { CHAINS("v_alignbit_b32 %0, %0, %1, %2") }
        if constexpr (OP == 6) { CHAINS("v_bfe_u32 %0, %0, 3, 16") }
        if constexpr (OP == 7) { CHAINS("v_pk_fma_f16 %0, %0, %1, %2") }
        if constexpr (OP == 8) { CHAINS("v_dot2_f32_f16 %0, %1, %2, %0") }
        if constexpr (OP == 9) { CHAINS("v_lshl_add_u32 %0, %0, 16, %1") }
        if constexpr (OP == 10) { CHAINS("v_pk_mul_lo_u16 %0, %0, %1") }
        if constexpr (OP == 11) { CHAINS("v_perm_b32 %0, %0, %1, %2") }
        if constexpr (OP == 12) { CHAINS("v_mul_hi_u32_u24 %0, %0, %1") }
        if constexpr (OP == 13) { CHAINS("v_dot4_u32_u8 %0, %1, %2, %0") }
        if constexpr (OP == 14) { CHAINS("v_lshrrev_b32 %0, 16, %0") }
        if constexpr (OP == 15) { CHAINS("v_mul_hi_u32 %0, %0, %1") }
    }
    uint32_t s = 0;
    for (int j = 0; j < 8; ++j) s += a[j];
    out[blockIdx.x * 256 + threadIdx.x] = s;
}

static const char* names[] = {"v_add_nc_u32", "v_mul_lo_u32", "v_mul_u32_u24", "v_mad_u32_u16", "v_sad_u8", "v_alignbit_b32", "v_bfe_u32",
                              "v_pk_fma_f16", "v_dot2_f32_f16", "v_lshl_add_u32", "v_pk_mul_lo_u16", "v_perm_b32", "v_mul_hi_u32_u24",
                              "v_dot4_u32_u8", "v_lshrrev_b32", "v_mul_hi_u32"};

static int sclk()
{
    FILE* f = fopen("/sys/class/drm/card0/device/pp_dpm_sclk", "r");
    if (!f) return -1;
    char line[128]; int mhz = -1;
    while (fgets(line, sizeof line, f)) if (strchr(line, '*')) { const char* c = strchr(line, ':'); if (c) mhz = atoi(c + 1); }
    fclose(f);
    return mhz;
}

template <int OP> static float run(uint32_t* out, int blocks, int n, int reps)
{
    hipEvent_t a, b;
    HC(hipEventCreate(&a)); HC(hipEventCreate(&b));
    for (int i = 0; i < 3; ++i) mb<OP><<<blocks, 256>>>(out, i, n);
    HC(hipEventRecord(a));
    for (int i = 0; i < reps; ++i) mb<OP><<<blocks, 256>>>(out, i, n);
    HC(hipEventRecord(b));
    HC(hipEventSynchronize(b));
    float ms; HC(hipEventElapsedTime(&ms, a, b));
    return ms / reps;
}

int main()
{
    const int blocks = 40 * 2 * 16 * 32 / 256 * 2, n = 2048, reps = 20;   // 2x the resident waves
    uint32_t* out; HC(hipMalloc(&out, blocks * 256 * 4));
    float t[16];
    for (int pass = 0; pass < 2; ++pass)
    {
        t[0] = run<0>(out, blocks, n, reps); t[1] = run<1>(out, blocks, n, reps); t[2] = run<2>(out, blocks, n, reps); t[3] = run<3>(out, blocks, n, reps);
        t[4] = run<4>(out, blocks, n, reps); t[5] = run<5>(out, blocks, n, reps); t[6] = run<6>(out, blocks, n, reps); t[7] = run<7>(out, blocks, n, reps);
        t[8] = run<8>(out, blocks, n, reps); t[9] = run<9>(out, blocks, n, reps); t[10] = run<10>(out, blocks, n, reps); t[11] = run<11>(out, blocks, n, reps);
        t[12] = run<12>(out, blocks, n, reps); t[13] = run<13>(out, blocks, n, reps); t[14] = run<14>(out, blocks, n, reps); t[15] = run<15>(out, blocks, n, reps);
        const int clk = sclk();
        // waves x n x 8 ops; SIMD-cycles per op at the sampled clock
        const double waves = blocks * 8.0, ops = waves * n * 8.0;
        for (int i = 0; i < 16; ++i)
            printf("{\"pass\":%d,\"op\":\"%s\",\"ms\":%.4f,\"rel_add\":%.3f,\"simd_cycles_per_op\":%.3f,\"sclk\":%d}\n", pass, names[i], t[i], t[i] / t[0],
                   t[i] * 1e-3 * clk * 1e6 * 80.0 / ops, clk);
    }
    return 0;
}
