// d2dcost: what an own-block device-to-device copy costs on the GPU stream (the copy an in-place gather saves).
// For each size: N back-to-back hipMemcpyDtoDAsync on one stream timed by events (GPU time per copy), and the same with
// a 1-thread kernel between copies (each copy then stands alone in the stream, as in a gather).
// usage: d2dcost [reps]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)
__global__ void tick(int *p) { if (threadIdx.x == 0) p[0] += 1; }
int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 200;
    const size_t sizes[] = {16, 20480, 122880, 614400, 1551360, 10485760, 20971520, 41943040, 104857600, 209715200};
    size_t maxb = 209715200;
    char *a, *b; int *t;
    CK(hipMalloc(&a, maxb)); CK(hipMalloc(&b, maxb)); CK(hipMalloc(&t, 64));
    CK(hipMemset(a, 1, maxb)); CK(hipMemset(b, 0, maxb));
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    for (size_t sz : sizes) {
        int n = sz >= (40u << 20) ? reps / 10 + 2 : reps;
        for (int mode = 0; mode < 3; mode++) {
            // mode 0: copies back to back; 1: a tiny kernel between copies; 2: the tiny kernels alone (baseline for 1)
            for (int w = 0; w < 3; w++) { if (mode != 2) CK(hipMemcpyDtoDAsync(b, a, sz, s)); tick<<<1, 64, 0, s>>>(t); }
            CK(hipStreamSynchronize(s));
            CK(hipEventRecord(e0, s));
            for (int i = 0; i < n; i++) {
                if (mode != 2) CK(hipMemcpyDtoDAsync(b, a, sz, s));
                if (mode != 0) tick<<<1, 64, 0, s>>>(t);
            }
            CK(hipEventRecord(e1, s));
            CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            const char *nm[] = {"copies", "copy+kernel", "kernel only"};
            printf("size=%zu mode=%s n=%d per=%.2f us  GB/s(r+w)=%.1f\n", sz, nm[mode], n, 1000.0 * ms / n,
                   mode == 2 ? 0.0 : 2.0 * sz * n / (ms * 1e6));
        }
    }
    return 0;
}
