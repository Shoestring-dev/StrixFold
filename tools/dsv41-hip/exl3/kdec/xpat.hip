// k-dec2 xpat: the decode experts' weight-read pattern alone (grouped_cp at R = 1, layer 39 shapes), to see what the read
// piece size costs. A place's matrix is [kt][ntiles][tw] (the checkpoint's tile order: a k row of tiles contiguous, k rows
// kt * ... apart), so a wave's k step reads one piece of CW 128-column blocks x 8 tiles x tw bytes (CW = 1: 768 B at K2 6,
// what grouped_cp reads) and its next step a k row further on. Variants: CW = 1, 2, 4 (fewer, wider waves: the same bytes
// in 1.5-3 KB pieces), and a flat stream of the same bytes. gate/up: 7 places x 2 matrices of 5120 x 1152 (kt 320, 9
// column blocks, SK 4); down: 7 x 1152 x 5120 (kt 72, 40 blocks, SK 1); 4 waves a block, each its k range, one step of
// loads in flight beyond the one in use (as the kernel). Timed as HIP graphs of 16 launches over rotating cold copies
// (back to back, served clocks), the median of 9 replays a launch.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s line %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)

struct Pat { const uint32_t* base; long long mat_words; int mats, kt, ntiles, tw_words, sk, cw; };

template <int MAXW>
__global__ void __launch_bounds__(128) xpat(Pat a, uint32_t* out) {
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int cbs = a.ntiles / 8 / a.cw;                       // column groups a matrix
    int b = blockIdx.x;
    const int s = b % a.sk; b /= a.sk;
    const int cb = b % cbs; b /= cbs;
    const int m = b;                                           // place x matrix
    const int per_warp = a.kt / (a.sk * 4);
    const int kt0 = s * per_warp * 4 + wave * per_warp;
    const int words = a.cw * 8 * a.tw_words;                   // a step's piece
    const int lw = words / 32;
    const uint32_t* mat = a.base + (long long)m * a.mat_words + (long long)cb * a.cw * 8 * a.tw_words;
    uint32_t pf[MAXW], acc = 0;
    for (int i = 0; i < MAXW; ++i) pf[i] = i < lw ? __builtin_nontemporal_load(mat + (long long)kt0 * a.ntiles * a.tw_words + i * 32 + lane) : 0u;
    for (int j = 0; j < per_warp; ++j) {
        uint32_t cur[MAXW];
        for (int i = 0; i < MAXW; ++i) cur[i] = pf[i];
        if (j + 1 < per_warp)
            for (int i = 0; i < MAXW; ++i)
                pf[i] = i < lw ? __builtin_nontemporal_load(mat + (long long)(kt0 + j + 1) * a.ntiles * a.tw_words + i * 32 + lane) : 0u;
        for (int i = 0; i < MAXW; ++i) acc ^= cur[i];
    }
    if (acc == 0x12345678u) out[0] = acc;
}

typedef uint32_t v4u __attribute__((ext_vector_type(4)));
__global__ void __launch_bounds__(256) flat(const v4u* p, size_t n16, uint32_t* out) {
    uint32_t acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) { const v4u v = __builtin_nontemporal_load(p + i); acc ^= v.x ^ v.w; }
    if (acc == 0x12345678u) out[0] = acc;
}

int main() {
    const size_t big = (size_t)1 << 30;
    uint32_t* buf; uint32_t* out;
    HC(hipMalloc(&buf, big)); HC(hipMalloc(&out, 64));
    HC(hipMemset(buf, 1, big));
    hipStream_t st; HC(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    hipEvent_t e0, e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    struct Shape { const char* name; int k, n, mats, sk; };
    for (int K : {3, 2}) {                                     // K2 6 (layer 39) and K2 4 (layers 18-22)
        const int tw_words = 16 * 16 * K / 32;                 // a tile's words (K bits a value)
        for (Shape sh : {Shape{"gate_up", 5120, 1152, 14, 4}, Shape{"down", 1152, 5120, 7, 1}}) {
            const int kt = sh.k / 16, ntiles = sh.n / 16;
            const long long mat_words = (long long)kt * ntiles * tw_words;
            const size_t launch_bytes = (size_t)mat_words * 4 * sh.mats;
            const int copies = std::min<int>(16, (int)(big / launch_bytes));
            auto time_graph = [&](auto&& launch) {
                hipGraph_t g; hipGraphExec_t ge;
                for (int c = 0; c < copies; ++c) launch(c);    // module load, first touch
                HC(hipStreamSynchronize(st));
                HC(hipStreamBeginCapture(st, hipStreamCaptureModeRelaxed));
                for (int c = 0; c < copies; ++c) launch(c);
                HC(hipStreamEndCapture(st, &g));
                HC(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
                std::vector<float> t;
                for (int r = 0; r < 11; ++r) {
                    HC(hipEventRecord(e0, st)); HC(hipGraphLaunch(ge, st)); HC(hipEventRecord(e1, st));
                    HC(hipEventSynchronize(e1));
                    float ms; HC(hipEventElapsedTime(&ms, e0, e1));
                    if (r >= 2) t.push_back(ms * 1e3f / copies);
                }
                std::sort(t.begin(), t.end());
                HC(hipGraphExecDestroy(ge)); HC(hipGraphDestroy(g));
                return t[t.size() / 2];
            };
            for (int cw : {1, 2, 3, 4}) {
                if ((ntiles / 8) % cw) continue;
                const int lw = cw * 8 * tw_words / 32;
                const int blocks = sh.mats * (ntiles / 8 / cw) * sh.sk;
                auto launch = [&](int c) {
                    Pat a{buf + (size_t)c * mat_words * sh.mats, mat_words, sh.mats, kt, ntiles, tw_words, sh.sk, cw};
                    if (lw <= 6) xpat<6><<<blocks, 128, 0, st>>>(a, out);
                    else if (lw <= 12) xpat<12><<<blocks, 128, 0, st>>>(a, out);
                    else xpat<24><<<blocks, 128, 0, st>>>(a, out);
                };
                const float us = time_graph(launch);
                printf("{\"k2\":%d,\"shape\":\"%s\",\"cw\":%d,\"piece_bytes\":%d,\"blocks\":%d,\"bytes\":%zu,\"us\":%.1f,\"gbs\":%.1f}\n",
                       2 * K, sh.name, cw, cw * 8 * tw_words * 4, blocks, launch_bytes, us, launch_bytes / (us * 1e3));
                fflush(stdout);
            }
            auto lflat = [&](int c) { flat<<<640, 256, 0, st>>>((const v4u*)(buf + (size_t)c * mat_words * sh.mats), launch_bytes / 16, out); };
            const float us = time_graph(lflat);
            printf("{\"k2\":%d,\"shape\":\"%s\",\"cw\":0,\"piece_bytes\":0,\"blocks\":640,\"bytes\":%zu,\"us\":%.1f,\"gbs\":%.1f}\n",
                   2 * K, sh.name, launch_bytes, us, launch_bytes / (us * 1e3));
            fflush(stdout);
        }
    }
    return 0;
}
