// k-dec2: does gfx1151's memory-side cache (MALL, 32 MB on Strix Halo) serve GPU weight reads, and can a touch on a
// side stream move a weight read off the critical path? Timing only.
// 1. hot vs cold: read S MB twice back to back (the second from whatever cache kept it) vs a cold read of S MB (after
//    256 MB of other reads), for S = 2..128 MB; plain and nontemporal loads (the decode kernels' weights are NT).
// 2. prefetch: a 1-block spin of T us on stream A (a gather's wait) with a touch of S MB on stream B, then the S MB read
//    on stream A after the spin (both joined): the read's own time against its cold time.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s line %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)
typedef uint32_t v4u __attribute__((ext_vector_type(4)));

template <bool NT>
__global__ void __launch_bounds__(256) rd(const v4u* __restrict__ p, size_t n16, uint32_t* out) {
    uint32_t acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n16; i += 4 * stride) {
        v4u v0, v1, v2, v3;
        if (NT) {
            v0 = __builtin_nontemporal_load(p + i); v1 = __builtin_nontemporal_load(p + i + stride);
            v2 = __builtin_nontemporal_load(p + i + 2 * stride); v3 = __builtin_nontemporal_load(p + i + 3 * stride);
        } else {
            v0 = p[i]; v1 = p[i + stride]; v2 = p[i + 2 * stride]; v3 = p[i + 3 * stride];
        }
        acc += (v0.x ^ v1.y) + (v2.z ^ v3.w);
    }
    for (; i < n16; i += stride) acc += p[i].x;
    if (acc == 0x12345678u) out[0] = acc;
}

__global__ void spin(long long ticks, uint32_t* out) {
    const long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {}
    if (threadIdx.x == 0) out[1] = 1;
}

static int sclk() {
    FILE* f = fopen("/sys/class/drm/card0/device/hwmon/hwmon3/freq1_input", "r");
    if (!f) return -1;
    long v = 0; if (fscanf(f, "%ld", &v) != 1) v = 0; fclose(f);
    return (int)(v / 1000000);
}

int main() {
    const size_t big = (size_t)1 << 30;  // 1 GiB: weights region (cold reads walk it) + a flush region
    char* p; uint32_t* out;
    HC(hipMalloc(&p, 2 * big)); HC(hipMalloc(&out, 64));
    HC(hipMemset(p, 1, 2 * big));
    hipStream_t sa, sb;
    HC(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking)); HC(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking));
    hipEvent_t e0, e1, ej;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1)); HC(hipEventCreate(&ej));
    int rate = 0; HC(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0));  // kHz
    const int blocks = 640;
    size_t cur = 0;   // next cold offset in the weights region
    auto flush = [&](hipStream_t s) { rd<false><<<blocks, 256, 0, s>>>((const v4u*)(p + big), (256u << 20) / 16, out); };
    auto timed = [&](hipStream_t s, const char* src, size_t bytes, bool nt) {
        HC(hipEventRecord(e0, s));
        if (nt) rd<true><<<blocks, 256, 0, s>>>((const v4u*)src, bytes / 16, out);
        else rd<false><<<blocks, 256, 0, s>>>((const v4u*)src, bytes / 16, out);
        HC(hipEventRecord(e1, s));
        HC(hipEventSynchronize(e1));
        float ms; HC(hipEventElapsedTime(&ms, e0, e1));
        return ms * 1e3;
    };
    // warm-up
    for (int i = 0; i < 3; ++i) flush(sa);
    HC(hipStreamSynchronize(sa));
    const int reps = 7;
    for (int mb : {2, 4, 8, 12, 16, 24, 32, 48, 64, 128})
        for (int nt = 0; nt < 2; ++nt) {
            std::vector<float> cold, hot;
            for (int r = 0; r < reps; ++r) {
                const size_t bytes = (size_t)mb << 20;
                if (cur + bytes > big) cur = 0;
                const char* src = p + cur; cur += bytes;
                flush(sa);
                cold.push_back(timed(sa, src, bytes, nt));
                hot.push_back(timed(sa, src, bytes, nt));
            }
            std::sort(cold.begin(), cold.end()); std::sort(hot.begin(), hot.end());
            const double c = cold[reps / 2], h = hot[reps / 2];
            printf("{\"test\":\"hotcold\",\"mb\":%d,\"nt\":%d,\"cold_us\":%.1f,\"hot_us\":%.1f,\"cold_gbs\":%.1f,\"hot_gbs\":%.1f,\"sclk\":%d}\n",
                   mb, nt, c, h, ((double)mb * 1048576) / (c * 1e3), ((double)mb * 1048576) / (h * 1e3), sclk());
            fflush(stdout);
        }
    // prefetch: spin T us on A while B touches S MB; then A reads the S MB (NT, as glinear) after joining B
    for (int mb : {4, 8, 16, 24}) for (int us : {50, 100, 200}) {
        std::vector<float> pre, cold;
        for (int r = 0; r < reps; ++r) {
            const size_t bytes = (size_t)mb << 20;
            if (cur + 2 * bytes > big) cur = 0;
            const char* src = p + cur; cur += bytes;
            const char* src2 = p + cur; cur += bytes;
            flush(sa); HC(hipStreamSynchronize(sa));
            spin<<<1, 32, 0, sa>>>((long long)rate * us / 1000, out);
            rd<false><<<blocks, 256, 0, sb>>>((const v4u*)src, bytes / 16, out);
            HC(hipEventRecord(ej, sb));
            HC(hipStreamWaitEvent(sa, ej, 0));
            pre.push_back(timed(sa, src, bytes, true));
            flush(sa);
            cold.push_back(timed(sa, src2, bytes, true));
        }
        std::sort(pre.begin(), pre.end()); std::sort(cold.begin(), cold.end());
        printf("{\"test\":\"prefetch\",\"mb\":%d,\"spin_us\":%d,\"after_touch_us\":%.1f,\"cold_us\":%.1f,\"sclk\":%d}\n", mb, us,
               pre[reps / 2], cold[reps / 2], sclk());
        fflush(stdout);
    }
    return 0;
}
