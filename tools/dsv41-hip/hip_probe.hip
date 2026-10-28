// hip_probe.hip [BYTES]: preflight's bounded, reversible HIP probe (handoff section 4).
//
// On device 0: the properties the port depends on, runtime and driver versions and hipMemGetInfo; then ONE allocation
// of BYTES (default 2,559,098,880 B, the plan's largest single device allocation: one layer's expert trellis buffer,
// weights.zig experts() -> DeviceBuffer.alloc), written by a kernel and checked by a kernel word for word; a 64 MiB
// slice copied to the host and checked there, complemented, copied back and checked on the device; then everything is
// freed and hipMemGetInfo read again. Prints one JSON object; exit status 0 only when every check passed and the
// runtime reports gfx1151. No timings: this is a functional probe, not a measurement.
//
// Build: hipcc --offload-arch=gfx1151 -O2 hip_probe.hip -o hip_probe. Run it under /var/tmp/halo-gpu.lock.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                                      \
  do {                                                                                                             \
    hipError_t e_ = (x);                                                                                           \
    if (e_ != hipSuccess) {                                                                                        \
      printf("{\"ok\": false, \"error\": \"%s: %s\", \"line\": %d}\n", #x, hipGetErrorString(e_), __LINE__);      \
      return 1;                                                                                                    \
    }                                                                                                              \
  } while (0)

__host__ __device__ static inline uint32_t pattern(size_t i, uint32_t seed) {
  return (uint32_t)(i * 2654435761u) ^ (uint32_t)(i >> 32) ^ seed;
}

__global__ void fill(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = pattern(i, seed);
}

// counts words of p[0, n) that differ from pattern(base + i, seed) ^ xorv
__global__ void check(const uint32_t* p, size_t n, size_t base, uint32_t seed, uint32_t xorv, unsigned long long* bad) {
  unsigned long long local = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    local += p[i] != (pattern(base + i, seed) ^ xorv);
  if (local) atomicAdd(bad, local);
}

int main(int argc, char** argv) {
  const size_t bytes = (argc > 1 ? strtoull(argv[1], nullptr, 10) : 2559098880ull) & ~(size_t)3;
  const uint32_t seed = 0x9e3779b9u;
  int count = 0;
  CK(hipGetDeviceCount(&count));
  CK(hipSetDevice(0));
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0));
  int runtime = 0, driver = 0;
  CK(hipRuntimeGetVersion(&runtime));
  CK(hipDriverGetVersion(&driver));
  size_t free0 = 0, total = 0, free1 = 0, free2 = 0, t = 0;
  CK(hipMemGetInfo(&free0, &total));

  uint32_t* d = nullptr;
  unsigned long long* bad = nullptr;
  CK(hipMalloc(&d, bytes));
  CK(hipMalloc(&bad, 2 * sizeof(unsigned long long)));
  CK(hipMemset(bad, 0, 2 * sizeof(unsigned long long)));
  CK(hipMemGetInfo(&free1, &t));
  const size_t words = bytes / 4;
  fill<<<1024, 256>>>(d, words, seed);
  CK(hipGetLastError());
  check<<<1024, 256>>>(d, words, 0, seed, 0u, bad);
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());

  // host round trip on the last 64 MiB (or the whole buffer if smaller)
  const size_t slice_words = (words < (16u << 20) ? words : (16u << 20));
  const size_t base = words - slice_words;
  std::vector<uint32_t> h(slice_words);
  CK(hipMemcpy(h.data(), d + base, slice_words * 4, hipMemcpyDeviceToHost));
  unsigned long long host_bad = 0;
  for (size_t i = 0; i < slice_words; i++) {
    host_bad += h[i] != pattern(base + i, seed);
    h[i] = ~h[i];
  }
  CK(hipMemcpy(d + base, h.data(), slice_words * 4, hipMemcpyHostToDevice));
  check<<<1024, 256>>>(d + base, slice_words, base, seed, 0xffffffffu, bad + 1);
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());
  unsigned long long nbad[2] = {0, 0};
  CK(hipMemcpy(nbad, bad, sizeof nbad, hipMemcpyDeviceToHost));
  CK(hipFree(d));
  CK(hipFree(bad));
  CK(hipMemGetInfo(&free2, &t));

  const bool arch_ok = strncmp(p.gcnArchName, "gfx1151", 7) == 0;
  const bool ok = arch_ok && nbad[0] == 0 && nbad[1] == 0 && host_bad == 0;
  printf("{\"ok\": %s, \"device_count\": %d, \"name\": \"%s\", \"gcnArchName\": \"%s\", \"arch_is_gfx1151\": %s,\n",
         ok ? "true" : "false", count, p.name, p.gcnArchName, arch_ok ? "true" : "false");
  printf(" \"multiProcessorCount\": %d, \"warpSize\": %d, \"totalGlobalMem\": %zu, \"integrated\": %d, "
         "\"managedMemory\": %d, \"pageableMemoryAccess\": %d, \"canMapHostMemory\": %d,\n",
         p.multiProcessorCount, p.warpSize, (size_t)p.totalGlobalMem, p.integrated, p.managedMemory,
         p.pageableMemoryAccess, p.canMapHostMemory);
  printf(" \"l2CacheSize\": %d, \"sharedMemPerBlock\": %zu, \"maxThreadsPerBlock\": %d, \"regsPerBlock\": %d, "
         "\"clockRate_khz\": %d, \"runtime_version\": %d, \"driver_version\": %d,\n",
         p.l2CacheSize, (size_t)p.sharedMemPerBlock, p.maxThreadsPerBlock, p.regsPerBlock, p.clockRate, runtime,
         driver);
  printf(" \"mem_get_info\": {\"total\": %zu, \"free_before\": %zu, \"free_while_held\": %zu, \"free_after\": %zu},\n",
         total, free0, free1, free2);
  printf(" \"allocation_bytes\": %zu, \"device_words_bad\": %llu, \"host_slice_words\": %zu, \"host_words_bad\": %llu, "
         "\"device_after_h2d_words_bad\": %llu}\n",
         bytes, nbad[0], slice_words, host_bad, nbad[1]);
  return ok ? 0 : 1;
}
