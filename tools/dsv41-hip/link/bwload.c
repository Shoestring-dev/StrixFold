// bwload: a known DRAM load for checking umcbw's scale. Copies a SIZE_MIB buffer into another PASSES times with
// non-temporal stores (no read-for-ownership), so each pass moves SIZE bytes read and SIZE bytes written, and prints
// the traffic it generated and the time it took.
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
int main(int argc, char **argv) {
    size_t mib = argc > 1 ? strtoul(argv[1], 0, 0) : 1024, passes = argc > 2 ? strtoul(argv[2], 0, 0) : 20;
    size_t n = mib << 20;
    __m256i *a = aligned_alloc(64, n), *b = aligned_alloc(64, n);
    if (!a || !b) return 1;
    memset(a, 1, n);
    memset(b, 2, n);
    double t0 = now();
    for (size_t p = 0; p < passes; p++) {
        for (size_t i = 0; i < n / 32; i++) _mm256_stream_si256(&b[i], _mm256_load_si256(&a[i]));
        _mm_sfence();
    }
    double dt = now() - t0;
    printf("bwload: %zu passes of %zu MiB in %.3f s: read %.2f GB, written %.2f GB, %.2f GB/s each way\n", passes, mib, dt,
           passes * n / 1e9, passes * n / 1e9, passes * n / 1e9 / dt);
    return 0;
}
