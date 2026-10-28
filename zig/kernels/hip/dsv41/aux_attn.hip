/* w-tri: DeepSeek-V4.1 sparse attention with a sink for gfx1151 (wave32 WMMA), replacing the Triton _sparse_attn_part
 * (kernels.py at 222a9c0b) on the decode / verify path (split keys; the Triton merge kernels combine the partials)
 * and on the prompt path (one pass, the output directly). ADAPTER.md section 6: group "aux", one by-value struct
 * (tf_ds_aux_attn_args, abi.h), registered in registry.json.
 *
 * What it computes, for each row r and head h (heads 32, head dim 512 a rank):
 *   keys: the window positions p-127 .. p that exist (a ring: slot = position % ring_size, plus the row's ring base;
 *   or linear rows from position wlo), then the row's picks idx[r, j] >= 0 (plus the row's compressed base), as bf16
 *   rows or packed FP4 (E2M1 codes: byte j = element j | element j + 256 << 4; E4M3 scale per 16 elements);
 *   s = (q . k) * scale (fp32), online softmax over the split's keys in chunks of 16 (m from -1e30, p = exp(s - m)),
 *   l += p (fp32), acc += bf16(p) . k (fp32: p rounded to bf16 before the PV product, as DeepSeek's and TensorFold's
 *   kernels do). Split mode writes m, l, acc in _sparse_attn_part's partial layout (base = (r * 2 + h / 16) * splits
 *   + split; m at base * 16 + h % 16); the one-pass mode writes bf16(acc / (l + exp(sink[h] - m))).
 * Key chunks: the window's 128 keys are chunks 0..7, the picks chunks 8..; split s takes chunks s, s + splits, ...
 * A row's results never depend on the other rows of the launch (row invariance), only on its own keys and splits.
 * Work: a workgroup a (row, split), all 32 heads (each key is read once a workgroup), 8 waves. Per chunk: S = Q K^T
 * by WMMA (wave w: heads 16 (w % 2), dims 128 (w / 2) of the 512 in two chains, its partial added in a fixed order)
 * and the keys transposed for PV | the softmax step (8 threads a head, DPP shuffles), then the next chunk's keys
 * staged in LDS as bf16 (FP4 decoded exactly through an fp16 byte table, the loads issued a chunk ahead, the picks two
 * chunks ahead) | acc += P K by WMMA (wave w: heads 16 (w / 4), dims 128 (w % 4): 8 accumulator tiles in registers):
 * three barriers a chunk. 42.7 KB of LDS and at most 6 waves a SIMD (the compiler holds VGPRs to that; 12 bytes of
 * scratch a lane, a few accesses a chunk): three workgroups a WGP.
 * Measured (2048 prompt rows, 512 FP4 picks, one pass): 4.9 ms vs 6.1 (two workgroups a WGP) and 8.2 (the scalar
 * FP4 decode, four barriers); the Triton kernel about 35 ms.
 * WMMA layouts (gfx11 wave32, v_wmma_f32_16x16x16_bf16; probed by t_hipattn.py): A: lane l holds row l % 16's 16 k;
 * B: lane l holds column l % 16's 16 k; D: lane l holds column l % 16, rows 2 i + l / 16 (element i). Lanes 16-31
 * repeat 0-15's A and B. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "abi.h"

typedef __bf16 tf_bf16x16 __attribute__((ext_vector_type(16)));
typedef float tf_f32x8 __attribute__((ext_vector_type(8)));

namespace {
constexpr int H = 32, HD = 512, CH = 16, WIN = 128, NWAVE = 8;

__device__ inline float bf16_to_f32(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }

// E2M1 magnitude code 0..7 -> 0, .5, 1, 1.5, 2, 3, 4, 6; bit 3 the sign
__device__ inline float e2m1(uint32_t c) {
    const uint32_t m = c & 7u;
    const uint32_t bits = m >= 2u ? ((((m >> 1) + 126u) << 23) | ((m & 1u) << 22)) : (m == 1u ? 0x3F000000u : 0u);
    return __uint_as_float(bits | ((c & 8u) << 28));
}

// E4M3 (fn) byte -> fp32, exactly (bias 7, subnormal step 2^-9, S.1111.111 NaN)
__device__ inline float e4m3(uint32_t c) {
    c &= 0xFFu;
    const uint32_t e = (c >> 3) & 15u, m = c & 7u;
    float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : float(m) * 0.001953125f;
    if ((c & 0x7Fu) == 0x7Fu) v = __uint_as_float(0x7FC00000u);
    return (c & 0x80u) ? -v : v;
}

// A workgroup barrier for LDS traffic only: this wave's LDS operations complete, then s_barrier. __syncthreads()
// (a workgroup release/acquire fence) also waits for every global load in flight (vmcnt 0), which would serialise the
// next chunk's key loads with this chunk's compute; those loads land in registers and are waited for where they are
// used. The memory clobber keeps the compiler from moving LDS accesses across it.
__device__ inline void lds_sync() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

__device__ inline tf_f32x8 wmma(tf_bf16x16 a, tf_bf16x16 b, tf_f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

typedef uint32_t tf_u32x4 __attribute__((ext_vector_type(4)));

// A device address from the argument struct as a global (address space 1) pointer: without it every access is a flat
// one, which also counts in lgkmcnt, so each LDS wait would wait for the next chunk's key loads too.
#define TF_GLOBAL __attribute__((address_space(1)))
template <class T> __device__ inline TF_GLOBAL T* gp(uint64_t a) { return (TF_GLOBAL T*)a; }

// 16 consecutive bf16 of global memory (32-byte aligned) as a fragment
__device__ inline tf_bf16x16 frag_row_g(const TF_GLOBAL __bf16* p) {
    const tf_u32x4 lo = *reinterpret_cast<const TF_GLOBAL tf_u32x4*>(p);
    const tf_u32x4 hi = *reinterpret_cast<const TF_GLOBAL tf_u32x4*>(p + 8);
    uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    tf_bf16x16 v;
    __builtin_memcpy(&v, w, 32);
    return v;
}

// 16 consecutive bf16 (32-byte aligned, LDS or global) as a fragment: two 16-byte loads, copied (memcpy idiom)
__device__ inline tf_bf16x16 frag_row(const __bf16* p) {
    const tf_u32x4 lo = *reinterpret_cast<const tf_u32x4*>(p);
    const tf_u32x4 hi = *reinterpret_cast<const tf_u32x4*>(p + 8);
    uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    tf_bf16x16 v;
    __builtin_memcpy(&v, w, 32);
    return v;
}

// a 32-byte row of 16 bf16 stored with its halves swapped when sw (the swz16 layout) as a fragment
__device__ inline tf_bf16x16 frag_swz(const __bf16* p, int sw) {
    const tf_u32x4 lo = *reinterpret_cast<const tf_u32x4*>(p + 8 * sw);
    const tf_u32x4 hi = *reinterpret_cast<const tf_u32x4*>(p + 8 * (sw ^ 1));
    uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    tf_bf16x16 v;
    __builtin_memcpy(&v, w, 32);
    return v;
}

__device__ inline uint32_t pack_bf16(float a, float b) {
    __bf16 h[2] = {(__bf16)a, (__bf16)b};
    uint32_t u;
    __builtin_memcpy(&u, h, 4);
    return u;
}
}  // namespace

// One staged key's 32 values of this thread (key t / 16, dims (t % 16) * 32 ..): the raw words, read a chunk ahead
// This thread's share of a staged key: 4 pieces of 8 values, piece u at element 8 (t % 16 + 16 u) (consecutive lanes
// hold consecutive 16-byte pieces: coalesced loads and conflict-free LDS stores)
struct KeyLoad {
    uint32_t w[16];      // bf16: piece u in w[4u .. 4u+3]; FP4: piece u's 8 code bytes in w[2u], w[2u+1]
    uint32_t s[4];       // FP4: piece u's E4M3 scale byte (raw: converted when staged, so loading never waits)
    int32_t kind;        // -1 none, 0 bf16 (window or compressed), 3 FP4
};

// v from lane ^ M of the same 16-lane row, by DPP (row_xmask), not an LDS-crossbar shuffle
template <int M> __device__ inline float xor16(float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 | M, 0xF, 0xF, false));
}

typedef _Float16 tf_h2 __attribute__((ext_vector_type(2)));

// Four FP4 codes (the low nibbles of c's bytes, element k in byte k) times an fp16 scale, as two words of bf16 pairs
// (elements 0, 1 and 2, 3). The E2M1 magnitude's fp16 high byte by a byte-permute table, the sign or-ed in; the
// products are exact in fp16 (at most 6 significant bits, 2^-10 .. 2688) and widen exactly to fp32, whose high halves
// are the bf16 bits.
__device__ inline void fp4x4_bf16(uint32_t c, tf_h2 sc2, uint32_t& lo, uint32_t& hi) {
    const uint32_t hb = __builtin_amdgcn_perm(0x46444240u, 0x3E3C3800u, c & 0x07070707u) | ((c & 0x08080808u) << 4);
    const uint32_t p01 = __builtin_amdgcn_perm(0u, hb, 0x010C000Cu), p23 = __builtin_amdgcn_perm(0u, hb, 0x030C020Cu);
    tf_h2 a, b;
    __builtin_memcpy(&a, &p01, 4);
    __builtin_memcpy(&b, &p23, 4);
    a = a * sc2;
    b = b * sc2;
    lo = __builtin_amdgcn_perm(__float_as_uint((float)a.y), __float_as_uint((float)a.x), 0x07060302u);
    hi = __builtin_amdgcn_perm(__float_as_uint((float)b.y), __float_as_uint((float)b.x), 0x07060302u);
}

// bf16 bits of an exact float (its low 16 bits are zero: an E2M1 value times an E4M3 scale has at most 7 significant
// bits), by truncation
__device__ inline uint32_t bf16_exact(float v) { return __float_as_uint(v) >> 16; }

// at most 6 waves a SIMD (VGPRs held to fit): three workgroups a WGP
extern "C" __global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6)))
void tf_ds_aux_sparse_attn_h32_d512(tf_ds_aux_attn_args a) {
    // the chunk's keys by key (S's B operand): rows padded by 16 bytes so the 16 lanes reading 16 rows at one column
    // hit different LDS banks
    __shared__ __attribute__((aligned(32))) __bf16 kc[CH][HD + 8];
    // the same keys by dimension (PV's B operand) and the chunk's p (PV's A operand): rows of 32 bytes whose two
    // 16-byte halves swap places when bit 2 of the row is set (swz16), so 8 lanes reading 8 rows hit different banks
    __shared__ __attribute__((aligned(32))) __bf16 kt[HD][CH];
    __shared__ __attribute__((aligned(32))) __bf16 pc[H][CH];
    __shared__ __attribute__((aligned(16))) float sp[4][2][16][16]; // S partials: dims quarter x head tile x head x key
    __shared__ float m_s[H], l_s[H], al_s[H];
    __shared__ int32_t kval[2][CH];                                // the chunk's key validity (double-buffered)
    // (42.7 KB in all: three workgroups a WGP)

    const int r = blockIdx.x, split = blockIdx.z, splits = gridDim.z;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const bool ring = a.flags & 1, has_comp = a.flags & 2, packed = a.flags & 4, final_ = a.flags & 8,
               has_base = a.flags & 16;
    const TF_GLOBAL __bf16* q = gp<const __bf16>(a.q) + (int64_t)r * H * HD;
    const int64_t p = gp<const int64_t>(a.pos)[r];
    const int n_idx = has_comp ? (a.n_idx < 512 ? a.n_idx : 512) : 0;
    const int nchunks = WIN / CH + (n_idx + CH - 1) / CH;
    const int64_t rbase = (ring && has_base) ? gp<const int64_t>(a.rbase)[r] : 0;
    const int64_t cbase = has_base && has_comp ? gp<const int64_t>(a.cbase)[r] : 0;
    const int64_t wlo = ring ? 0 : gp<const int64_t>(a.wlo)[0];
    if (tid < H) {
        m_s[tid] = -1e30f;
        l_s[tid] = 0.f;
    }
    // this wave's S work: heads 16 s_mt, dims 128 s_kq; its 8 Q fragments stay in registers for every chunk
    const int s_mt = w & 1, s_kq = w >> 1;
    tf_bf16x16 qf[8];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) qf[ks] = frag_row_g(q + (s_mt * 16 + l16) * HD + s_kq * 128 + ks * 16);
    const int pv_mt = w >> 2, pv_n0 = (w & 3) * 8;    // PV tiles of this wave: heads 16 pv_mt, dim tiles pv_n0 ..
    tf_f32x8 acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = tf_f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    __syncthreads();

    const int sn = tid >> 4, sj = tid & 15;          // the staged key and the piece column of this thread
    // the source of key n of chunk c: window slot row or pool row (-1 none), and its kind
    int64_t pk = -1;                                    // the raw pick of the next chunk to load (idx_of)
    auto source = [&](int c, int n, int32_t& kind) -> int64_t {
        if (c < WIN / CH) {
            kind = 0;
            const int64_t wp = p - (WIN - 1) + c * CH + n;
            if (wp < 0) return -1;
            if (ring) return wp % a.ring_size + rbase;
            const int64_t s = wp - wlo;
            return s >= 0 ? s : -1;
        }
        kind = packed ? 3 : 2;
        return pk >= 0 ? pk + cbase : -1;                 // the pick, loaded a chunk earlier (idx_of)
    };
    // the raw pick of key sn of chunk c (-1 none), loaded a chunk before the keys are: the keys' addresses never
    // wait for it
    auto idx_of = [&](int c) -> int64_t {
        if (c < WIN / CH || c >= nchunks) return -1;
        const int t = (c - WIN / CH) * CH + sn;
        return t < n_idx ? gp<const int64_t>(a.idx)[(int64_t)r * a.n_idx + t] : -1;
    };
    auto load = [&](int c, KeyLoad& kl) {
        int32_t kind;
        const int64_t row = source(c, sn, kind);
        kl.kind = row < 0 ? -1 : (kind == 3 ? 3 : 0);
        if (row < 0) return;
        if (kind == 3) {
            const TF_GLOBAL uint8_t* codes = gp<const uint8_t>(a.comp) + row * (HD / 2);
            const TF_GLOBAL uint8_t* scales = gp<const uint8_t>(a.csc) + row * (HD / 16);
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                // code bytes 8 (sj + 16 u) ..: elements 8 (sj + 16 u) .. (low nibbles, piece u) and + 256 (high, u + 2)
                typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
                const u32x2 x = *reinterpret_cast<const TF_GLOBAL u32x2*>(codes + 8 * (sj + 16 * u));
                kl.w[2 * u] = x.x;
                kl.w[2 * u + 1] = x.y;
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) kl.s[u] = scales[(sj + 16 * u) / 2];
        } else {
            const TF_GLOBAL tf_u32x4* src = reinterpret_cast<const TF_GLOBAL tf_u32x4*>(
                gp<const __bf16>(kind ? a.comp : a.wsrc) + row * HD);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const tf_u32x4 x = src[sj + 16 * u];
                kl.w[4 * u] = x.x; kl.w[4 * u + 1] = x.y; kl.w[4 * u + 2] = x.z; kl.w[4 * u + 3] = x.w;
            }
        }
    };
    KeyLoad nxt;
    // stage the keys in nxt (loaded a chunk earlier) into kc as bf16, and their validity into kval[i & 1]
    auto stage = [&](int i) {
        const KeyLoad cur = nxt;
        uint32_t wd[16];
        if (cur.kind < 0) {
#pragma unroll
            for (int k = 0; k < 16; ++k) wd[k] = 0u;
        } else if (cur.kind == 3) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int sh = u >= 2 ? 4 : 0;                     // elements 256..511: the high nibbles
                const _Float16 sc = (_Float16)e4m3(cur.s[u]);      // exact (E4M3 is a subset of fp16)
                const tf_h2 sc2 = {sc, sc};
#pragma unroll
                for (int v = 0; v < 2; ++v)
                    fp4x4_bf16(cur.w[2 * (u & 1) + v] >> sh, sc2, wd[4 * u + 2 * v], wd[4 * u + 2 * v + 1]);
            }
        } else {
#pragma unroll
            for (int k = 0; k < 16; ++k) wd[k] = cur.w[k];
        }
        if ((tid & 15) == 0) kval[i & 1][sn] = cur.kind >= 0;
        tf_u32x4* dst = reinterpret_cast<tf_u32x4*>(&kc[sn][0]);
#pragma unroll
        for (int u = 0; u < 4; ++u) dst[sj + 16 * u] = tf_u32x4{wd[4 * u], wd[4 * u + 1], wd[4 * u + 2], wd[4 * u + 3]};
    };
    if (split < nchunks) {
        pk = idx_of(split);
        load(split, nxt);
        pk = idx_of(split + splits);
        stage(0);
        if (split + splits < nchunks) load(split + splits, nxt);
        pk = idx_of(split + 2 * splits);
    }
    lds_sync();

    // Three barriers a chunk: (S, transpose) | (softmax, stage the next chunk: kc is free once read) | PV
    for (int c = split, it = 0; c < nchunks; c += splits, ++it) {
        // 1. S partials: wave w: heads 16 s_mt, dims 128 s_kq (two independent WMMA chains: even and odd k-steps);
        // and the keys transposed into kt (thread t: dims t and t + 256, 16 keys each: column reads, two row stores)
        {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int d = tid + half * 256;
                uint32_t col[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    __bf16 two[2] = {kc[2 * k][d], kc[2 * k + 1][d]};
                    __builtin_memcpy(&col[k], two, 4);
                }
                tf_u32x4* kr = reinterpret_cast<tf_u32x4*>(&kt[d][0]);
                const int sw = (d >> 2) & 1;
                kr[sw] = tf_u32x4{col[0], col[1], col[2], col[3]};
                kr[sw ^ 1] = tf_u32x4{col[4], col[5], col[6], col[7]};
            }
            tf_f32x8 s0 = {0, 0, 0, 0, 0, 0, 0, 0}, s1 = s0;
#pragma unroll
            for (int ks = 0; ks < 8; ks += 2) {
                s0 = wmma(qf[ks], frag_row(&kc[l16][s_kq * 128 + ks * 16]), s0);
                s1 = wmma(qf[ks + 1], frag_row(&kc[l16][s_kq * 128 + ks * 16 + 16]), s1);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) sp[s_kq][s_mt][2 * i + hi][l16] = s0[i] + s1[i];
        }
        lds_sync();
        // 2. the softmax step: 8 threads a head, 2 keys a thread; max and sum over the 8 by DPP xor shuffles; then
        // the next chunk's keys staged into kc and the one after it loaded
        {
            const int h = tid >> 3, g = tid & 7, mt = h >> 4, hr = h & 15;
            typedef float f2 __attribute__((ext_vector_type(2)));
            f2 sp2[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) sp2[k] = *reinterpret_cast<const f2*>(&sp[k][mt][hr][2 * g]);
            float sv[2];
            const float m_old = m_s[h];
            float mx = m_old;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int n = 2 * g + u;
                const float d = ((sp2[0][u] + sp2[1][u]) + sp2[2][u]) + sp2[3][u];
                sv[u] = kval[it & 1][n] ? d * a.scale : -__builtin_inff();
                mx = fmaxf(mx, sv[u]);
            }
            mx = fmaxf(mx, xor16<1>(mx));
            mx = fmaxf(mx, xor16<2>(mx));
            mx = fmaxf(mx, xor16<4>(mx));
            const float p0 = expf(sv[0] - mx), p1 = expf(sv[1] - mx);
            float sum = p0 + p1;
            sum += xor16<1>(sum);
            sum += xor16<2>(sum);
            sum += xor16<4>(sum);
            reinterpret_cast<uint32_t*>(&pc[h][0])[(((g >> 2) ^ ((h >> 2) & 1)) << 2) | (g & 3)] = pack_bf16(p0, p1);
            if (g == 0) {
                const float alpha = expf(m_old - mx);
                l_s[h] = l_s[h] * alpha + sum;
                m_s[h] = mx;
                al_s[h] = alpha;
            }
        }
        if (c + splits < nchunks) {
            stage(it + 1);
            if (c + 2 * splits < nchunks) load(c + 2 * splits, nxt);
            pk = idx_of(c + 3 * splits);
        }
        lds_sync();
        // 3. acc = acc * alpha + P K: wave w: heads 16 pv_mt, dim tiles pv_n0 .. + 8
        {
            const tf_bf16x16 fa = frag_swz(&pc[pv_mt * 16 + l16][0], (l16 >> 2) & 1);
            float al[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) al[i] = al_s[pv_mt * 16 + 2 * i + hi];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[j][i] *= al[i];
                acc[j] = wmma(fa, frag_swz(&kt[(pv_n0 + j) * 16 + l16][0], (l16 >> 2) & 1), acc[j]);
            }
        }
        lds_sync();
    }

    // the split's state, or the one-pass output
    if (final_) {
        const TF_GLOBAL float* sink = gp<const float>(a.sink);
        TF_GLOBAL __bf16* out = gp<__bf16>(a.out) + (int64_t)r * H * HD;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int h = pv_mt * 16 + 2 * i + hi;
            const float den = l_s[h] + expf(sink[h] - m_s[h]);
#pragma unroll
            for (int j = 0; j < 8; ++j) out[h * HD + (pv_n0 + j) * 16 + l16] = (__bf16)(acc[j][i] / den);
        }
        return;
    }
    TF_GLOBAL float* pm = gp<float>(a.pm);
    TF_GLOBAL float* pl = gp<float>(a.pl);
    TF_GLOBAL float* po = gp<float>(a.po);
    if (tid < H) {
        const int64_t base = ((int64_t)r * (H / 16) + tid / 16) * splits + split;
        pm[base * 16 + tid % 16] = m_s[tid];
        pl[base * 16 + tid % 16] = l_s[tid];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int h = pv_mt * 16 + 2 * i + hi;
        const int64_t base = ((int64_t)r * (H / 16) + h / 16) * splits + split;
#pragma unroll
        for (int j = 0; j < 8; ++j) po[(base * 16 + h % 16) * HD + (pv_n0 + j) * 16 + l16] = acc[j][i];
    }
}
