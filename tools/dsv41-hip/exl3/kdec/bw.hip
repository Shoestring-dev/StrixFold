// k-dec: read bandwidth of a plain streaming kernel on gfx1151 against load width and loads in flight (what a decode
// kernel's weight stream could reach). 1 GiB buffer, each wave reads contiguous chunks; sum to defeat dead-code removal.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s line %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)
typedef uint32_t v4u __attribute__((ext_vector_type(4)));

// WIDTH 4 or 16 bytes a lane a load; UNR loads in flight a wave before using them; nontemporal like the kernels
template <int WIDTH, int UNR, bool NT>
__global__ void __launch_bounds__(128) rd(const uint32_t* __restrict__ p, size_t words, uint32_t* out, int chunk_words) {
    const int lane = threadIdx.x & 31;
    const size_t wave = (size_t)blockIdx.x * 4 + (threadIdx.x >> 5);
    const size_t nw = (size_t)gridDim.x * 4;
    uint32_t acc = 0;
    for (size_t c = wave; (c + 1) * chunk_words <= words; c += nw) {
        const uint32_t* q = p + c * chunk_words;
        if constexpr (WIDTH == 4) {
            for (int i = 0; i < chunk_words / 32; i += UNR) {
                uint32_t v[UNR];
#pragma unroll
                for (int u = 0; u < UNR; ++u) v[u] = NT ? __builtin_nontemporal_load(q + (i + u) * 32 + lane) : q[(i + u) * 32 + lane];
#pragma unroll
                for (int u = 0; u < UNR; ++u) acc += v[u];
            }
        } else {
            for (int i = 0; i < chunk_words / 128; i += UNR) {
                v4u v[UNR];
#pragma unroll
                for (int u = 0; u < UNR; ++u) {
                    const v4u* a = reinterpret_cast<const v4u*>(q + (i + u) * 128 + 4 * lane);
                    v[u] = NT ? __builtin_nontemporal_load(a) : *a;
                }
#pragma unroll
                for (int u = 0; u < UNR; ++u) acc += v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
            }
        }
    }
    if (acc == 0x12345678u) out[0] = acc;
}

static int sclk()
{
    FILE* f = fopen("/sys/class/drm/card0/device/pp_dpm_sclk", "r");
    if (!f) return -1;
    char line[128]; int mhz = -1;
    while (fgets(line, sizeof line, f)) if (strchr(line, '*')) { const char* c = strchr(line, ':'); if (c) mhz = atoi(c + 1); }
    fclose(f);
    return mhz;
}

template <int W, int U, bool NT> static void run(const uint32_t* p, size_t words, uint32_t* out, int blocks, int chunk, const char* name)
{
    hipEvent_t a, b;
    HC(hipEventCreate(&a)); HC(hipEventCreate(&b));
    for (int i = 0; i < 3; ++i) rd<W, U, NT><<<blocks, 128>>>(p, words, out, chunk);
    HC(hipEventRecord(a));
    const int reps = 10;
    for (int i = 0; i < reps; ++i) rd<W, U, NT><<<blocks, 128>>>(p, words, out, chunk);
    HC(hipEventRecord(b));
    HC(hipEventSynchronize(b));
    const int clk = sclk();
    float ms; HC(hipEventElapsedTime(&ms, a, b));
    const double bytes = (double) (words / chunk) * chunk * 4.0;
    printf("{\"kernel\":\"%s\",\"width\":%d,\"inflight\":%d,\"nt\":%d,\"blocks\":%d,\"chunk_bytes\":%d,\"gbs\":%.1f,\"sclk_after\":%d}\n", name, W, U, NT ? 1 : 0, blocks, chunk * 4, bytes * reps / (ms * 1e6), clk);
    fflush(stdout);
}

int main()
{
    const size_t words = (size_t) 1 << 28;   // 1 GiB
    uint32_t *p, *out;
    HC(hipMalloc(&p, words * 4)); HC(hipMalloc(&out, 64));
    HC(hipMemset(p, 1, words * 4));
    for (int blocks : {480, 960, 1920})
        for (int chunk : {1536, 12288})       // a decode k step's 8 tiles at K2 6 (6 KB is 4 steps), and 48 KB
        {
            run<4, 6, true>(p, words, out, blocks, chunk, "b32");
            run<4, 12, true>(p, words, out, blocks, chunk, "b32");
            run<16, 2, true>(p, words, out, blocks, chunk, "b128");
            run<16, 4, true>(p, words, out, blocks, chunk, "b128");
            run<16, 4, false>(p, words, out, blocks, chunk, "b128");
        }
    return 0;
}
