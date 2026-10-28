// k-dec2: does a forked branch of a captured HIP graph run beside the main branch on gfx1151 (the engine's prefetch
// stream: fork after a kernel, touch the next weights, join before their consumer)? Each graph, replayed with the
// weights cold (256 MB of other reads between replays, outside the timing):
//   none:   main: spin T us (one kernel, or KS kernels of T / KS us), then the consumer's NT read of S MB
//   fork:   the same main branch; a forked branch touches the S MB (plain loads) beside it; joined before the consumer
//   serial: spin, touch, consumer, all on the main stream
// fork ~ max(T, touch) + the consumer's hot read: the branch ran beside; fork ~ serial: it was serialized.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
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

int main() {
    const size_t big = (size_t)1 << 30;
    char* p; uint32_t* out;
    HC(hipMalloc(&p, 2 * big)); HC(hipMalloc(&out, 64));
    HC(hipMemset(p, 1, 2 * big));
    hipStream_t sa, sb;
    HC(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking)); HC(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking));
    hipEvent_t ef, ej, t0, t1;
    HC(hipEventCreateWithFlags(&ef, hipEventDisableTiming)); HC(hipEventCreateWithFlags(&ej, hipEventDisableTiming));
    HC(hipEventCreate(&t0)); HC(hipEventCreate(&t1));
    int rate = 0; HC(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0));  // kHz
    const int blocks = 640, tblocks = getenv("TB") ? atoi(getenv("TB")) : 80;
    auto flush = [&]() { rd<false><<<blocks, 256, 0, sa>>>((const v4u*)(p + big), (256u << 20) / 16, out); };
    const int reps = 9;
    for (int mb : {8, 16, 24}) for (int us : {50, 100, 200}) for (int ks : {1, 10}) {
        const size_t bytes = (size_t)mb << 20;
        float res[3];
        for (int kind = 0; kind < 3; ++kind) {   // none, fork, serial
            const char* src = p + (size_t)kind * (64u << 20);
            hipGraph_t g; hipGraphExec_t ge;
            HC(hipStreamBeginCapture(sa, hipStreamCaptureModeRelaxed));
            if (kind == 1) { HC(hipEventRecord(ef, sa)); HC(hipStreamWaitEvent(sb, ef, 0)); }
            for (int k = 0; k < ks; ++k) spin<<<1, 32, 0, sa>>>((long long)rate * us / 1000 / ks, out);
            if (kind == 1) {
                rd<false><<<tblocks, 256, 0, sb>>>((const v4u*)src, bytes / 16, out);
                HC(hipEventRecord(ej, sb)); HC(hipStreamWaitEvent(sa, ej, 0));
            }
            if (kind == 2) rd<false><<<tblocks, 256, 0, sa>>>((const v4u*)src, bytes / 16, out);
            rd<true><<<blocks, 256, 0, sa>>>((const v4u*)src, bytes / 16, out);
            HC(hipStreamEndCapture(sa, &g));
            HC(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
            std::vector<float> t;
            for (int r = 0; r < reps + 2; ++r) {
                flush();
                HC(hipEventRecord(t0, sa));
                HC(hipGraphLaunch(ge, sa));
                HC(hipEventRecord(t1, sa));
                HC(hipEventSynchronize(t1));
                float ms; HC(hipEventElapsedTime(&ms, t0, t1));
                if (r >= 2) t.push_back(ms * 1e3f);
            }
            std::sort(t.begin(), t.end());
            res[kind] = t[reps / 2];
            HC(hipGraphExecDestroy(ge)); HC(hipGraphDestroy(g));
        }
        printf("{\"test\":\"graph_fork\",\"mb\":%d,\"spin_us\":%d,\"spin_kernels\":%d,\"touch_blocks\":%d,\"none_us\":%.1f,\"fork_us\":%.1f,\"serial_us\":%.1f}\n",
               mb, us, ks, tblocks, res[0], res[1], res[2]);
        fflush(stdout);
    }
    return 0;
}
