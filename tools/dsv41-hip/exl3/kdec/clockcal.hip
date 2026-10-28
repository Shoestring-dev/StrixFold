// clockcal: which host clock rocprofv3's kernel timestamps are in. Five ~2 ms kernels, each bracketed by
// clock_gettime of BOOTTIME / MONOTONIC / MONOTONIC_RAW before the launch and after hipDeviceSynchronize. Run under
// rocprofv3 --kernel-trace; a clock is the trace's when every kernel's [start, end] lies inside its bracket.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <ctime>
__global__ void spin(long long cycles, int* out)
{
    long long t0 = wall_clock64();
    while (wall_clock64() - t0 < cycles) {}
    if (threadIdx.x == 0) out[blockIdx.x] = 1;
}
static long long ns(clockid_t c) { timespec t; clock_gettime(c, &t); return t.tv_sec * 1000000000LL + t.tv_nsec; }
int main()
{
    int* d; hipMalloc(&d, 1024);
    int rate = 0; hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0);  // kHz
    long long cyc = (long long) rate * 2;  // ~2 ms
    spin<<<1, 64>>>(cyc / 10, d); hipDeviceSynchronize();
    for (int i = 0; i < 5; ++i)
    {
        long long b0 = ns(CLOCK_BOOTTIME), m0 = ns(CLOCK_MONOTONIC), r0 = ns(CLOCK_MONOTONIC_RAW);
        spin<<<1, 64>>>(cyc, d);
        hipDeviceSynchronize();
        long long b1 = ns(CLOCK_BOOTTIME), m1 = ns(CLOCK_MONOTONIC), r1 = ns(CLOCK_MONOTONIC_RAW);
        printf("{\"i\": %d, \"boot\": [%lld, %lld], \"mono\": [%lld, %lld], \"raw\": [%lld, %lld]}\n", i, b0, b1, m0, m1, r0, r1);
    }
    return 0;
}
