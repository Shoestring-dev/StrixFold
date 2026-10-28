/* w-link's collectives proxy kernels (tools/dsv41-hip/COLLECTIVES-DESIGN.md section 2.5): the GPU half of tcp_proxy,
 * the all-gather that can live inside a HIP graph. Per gather, on the model stream:
 *   stage   copies this rank's block into its transmit slot of a mapped coherent slab and into its own place in recv,
 *           then publishes the slot (`ready = seq`) for the host proxy thread, which sends it over TCP and receives
 *           the peer's block into the slot's receive half (`done = seq`);
 *   collect waits for `done == seq` (one guard load per block, never per thread), copies the peer's block into recv,
 *           and releases the slot (`consumed = seq`).
 * The sequence number lives in the slab and is advanced by the stage kernel itself, so a replayed graph runs fresh
 * gathers; the size and site of each gather are written into its slot for the proxy, which never sees the host call.
 * Every wait is bounded (wall_clock64 ticks): on timeout or a set abort word the kernel sets abort and returns without
 * touching the data, and the host proxy fails the collective. Ordering: the copies end with __threadfence_system and
 * the flags are system-scope release stores / acquire loads on fine-grained (coherent) host memory, as the gpuvis
 * Phase-0 test validated on gfx1151 (0 element errors over 161,930 exchanges per rank, slots reused >= 1,275 times). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "abi.h"

// the slab header: tf_ds_coll_shared / tf_ds_coll_slot, one definition in abi.h (checked against tcp_proxy.zig)

__device__ __forceinline__ bool tf_ds_coll_aborted(tf_ds_coll_shared* s) {
    return __hip_atomic_load(&s->abort, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != 0;
}

// 2 only over 0: the host's 1 (its poison, close) stays, so the proxy's log tells a device wait that gave up from the
// host's abort (tcp_proxy.zig deviceGaveUp)
__device__ __forceinline__ void tf_ds_coll_fail(tf_ds_coll_shared* s) {
    uint32_t idle = 0u;
    __hip_atomic_compare_exchange_strong(&s->abort, &idle, 2u, __ATOMIC_RELEASE, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Copies `bytes` (any size; 16-byte words where both ends allow it) with the whole grid.
__device__ __forceinline__ void tf_ds_coll_copy(uint8_t* dst, const uint8_t* src, uint64_t bytes) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t step = (uint64_t)gridDim.x * blockDim.x;
    if ((((uintptr_t)dst | (uintptr_t)src) & 15) == 0) {
        const uint64_t n16 = bytes / 16;
        const uint4* s4 = reinterpret_cast<const uint4*>(src);
        uint4* d4 = reinterpret_cast<uint4*>(dst);
        for (uint64_t i = tid; i < n16; i += step) d4[i] = s4[i];
        for (uint64_t i = n16 * 16 + tid; i < bytes; i += step) dst[i] = src[i];
    } else {
        for (uint64_t i = tid; i < bytes; i += step) dst[i] = src[i];
    }
}

/* stage: tx = slab transmit slots (nslot x slot_bytes), own = recv + rank * bytes (skipped when equal to send). */
extern "C" __global__ void __launch_bounds__(256) tf_ds_coll_stage(tf_ds_coll_shared* s, const uint8_t* send,
        uint8_t* tx, uint8_t* own, uint64_t bytes, uint64_t slot_bytes, uint32_t nslot, uint32_t site,
        uint64_t deadline_ticks) {
    __shared__ uint64_t q_s;
    __shared__ int go;
    if (threadIdx.x == 0) {
        go = 0;
        const uint64_t q = __hip_atomic_load(&s->seq, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) + 1;
        q_s = q;
        const uint32_t slot = (uint32_t)(q % nslot);
        // the slot is free once the gather nslot earlier through it was collected
        const uint64_t start = wall_clock64();
        while (!tf_ds_coll_aborted(s)) {
            const uint64_t c = __hip_atomic_load(&s->slot[slot].consumed, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
            if (q <= nslot || c + nslot >= q) { go = 1; break; }
            if (wall_clock64() - start >= deadline_ticks) { tf_ds_coll_fail(s); break; }
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    if (!go) return;
    const uint64_t q = q_s;
    const uint32_t slot = (uint32_t)(q % nslot);
    tf_ds_coll_copy(tx + (uint64_t)slot * slot_bytes, send, bytes);
    if (own != send) tf_ds_coll_copy(own, send, bytes);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t n = __hip_atomic_fetch_add(&s->staged, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1;
        if (n == gridDim.x) {
            __hip_atomic_store(&s->staged, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            s->slot[slot].bytes = bytes;
            s->slot[slot].site = site;
            __hip_atomic_store(&s->seq, q, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(&s->slot[slot].ready, q, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

/* collect: rx = slab receive slots (nslot x slot_bytes), peer = recv + peer_rank * bytes. */
extern "C" __global__ void __launch_bounds__(256) tf_ds_coll_collect(tf_ds_coll_shared* s, const uint8_t* rx,
        uint8_t* peer, uint64_t bytes, uint64_t slot_bytes, uint32_t nslot, uint64_t deadline_ticks) {
    __shared__ int go;
    __shared__ uint64_t q_s;
    if (threadIdx.x == 0) {
        go = 0;
        const uint64_t q = __hip_atomic_load(&s->seq, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        q_s = q;
        const uint32_t slot = (uint32_t)(q % nslot);
        const uint64_t start = wall_clock64();
        while (!tf_ds_coll_aborted(s)) {
            const uint64_t d = __hip_atomic_load(&s->slot[slot].done, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
            if (d == q) { go = 1; break; }
            if (d > q || wall_clock64() - start >= deadline_ticks) { tf_ds_coll_fail(s); break; }
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    if (!go) return;
    const uint64_t q = q_s;
    const uint32_t slot = (uint32_t)(q % nslot);
    tf_ds_coll_copy(peer, rx + (uint64_t)slot * slot_bytes, bytes);
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t n = __hip_atomic_fetch_add(&s->collected, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1;
        if (n == gridDim.x) {
            __hip_atomic_store(&s->collected, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&s->slot[slot].consumed, q, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}
