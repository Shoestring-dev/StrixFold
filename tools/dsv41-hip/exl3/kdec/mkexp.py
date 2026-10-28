#!/usr/bin/env python3
"""mkexp.py SRC OUT EXP[+EXP...]: experiment variants of exl3.hip for timing only (never committed).
only1:  the decode kernels as 1-row-only kernels (no multi-row / wide paths; grouped_cp's red one row): what a lean
        R == 1 entry point would gain from its occupancy (applies to the row-invariant source before the lean entry
        points existed).
pkhash: the MUL1 hash without v_mul_lo_u32 (3.4 issue slots each on gfx1151): both values' high products in one
        v_pk_mul_lo_u16, the low products and the sums by v_mad_u32_u16 (16-bit sources: no masking).
mul24:  w-exl3's EXL3_MUL24 (two 24-bit multiplies and a shift-add a value)."""
import sys

src, out = sys.argv[1], sys.argv[2]
exps = sys.argv[3].split("+")
s = open(src).read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a, s.count(a))
    s = s.replace(a, b)


OLD_PAIR = """    const uint32_t w0 = u >> 16;
    const uint32_t w1 = __builtin_amdgcn_ubfe(u, 16 - D, 16);
    const uint32_t s0 = __builtin_amdgcn_sad_u8(mul1_hash(w0), 0u, 0x64006400u);   // low half: 0x6400 + bytesum
    const uint32_t s = __builtin_amdgcn_sad_hi_u8(mul1_hash(w1), 0u, s0);           // high half: the second's"""

PK_PAIR = """    // W = w0 (u's high half) : w1 (u >> (16 - D), low half); B = (w * 0x83DC) mod 2^16 for both halves; then
    // h = w * 0xD12D + (B << 16) = w * 0x83DCD12D mod 2^32 by v_mad_u32_u16 (it reads only the 16-bit halves)
    const uint32_t W = __builtin_amdgcn_perm(u, u >> (16 - D), 0x07060100u);
    uint32_t B, h0, h1;
    asm("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(B) : "v"(W), "v"(0x83DC83DCu));
    asm("v_mad_u32_u16 %0, %1, %2, %3 op_sel:[1,0,0,0]" : "=v"(h0) : "v"(W), "v"(0xD12Du), "v"(B & 0xffff0000u));
    asm("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(h1) : "v"(W), "v"(0xD12Du), "v"(B << 16));
    const uint32_t s0 = __builtin_amdgcn_sad_u8(h0, 0u, 0x64006400u);   // low half: 0x6400 + bytesum
    const uint32_t s = __builtin_amdgcn_sad_hi_u8(h1, 0u, s0);          // high half: the second's"""

OLD_STRIP = """    uint32_t pf[LW];
    uint4 xa;
    {
        const uint32_t* p = tiles + (int64_t)kt0 * step;
#pragma unroll
        for (int i = 0; i < LW; ++i) pf[i] = __builtin_nontemporal_load(gl(p) + i * 32 + lane);
        xa = *gl(reinterpret_cast<const uint4*>(xrow + 16 * kt0 + 8 * h));
    }
    for (int j = 0; j < n; ++j) {
#pragma unroll
        for (int i = 0; i < LW; ++i) st[i * 32 + lane] = pf[i];
        const uint4 x = xa;
        if (j + 1 < n) {
            const uint32_t* p = tiles + (int64_t)(kt0 + j + 1) * step;
#pragma unroll
            for (int i = 0; i < LW; ++i) pf[i] = __builtin_nontemporal_load(gl(p) + i * 32 + lane);
            xa = *gl(reinterpret_cast<const uint4*>(xrow + 16 * (kt0 + j + 1) + 8 * h));
        }
        lds_fence();
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            uint32_t own[4];
            decode_own<K2>(st + t * TW, win, own);
            float a = dot2(own[0], x.x, s[t]);
            a = dot2(own[1], x.y, a);
            a = dot2(own[2], x.z, a);
            s[t] = dot2(own[3], x.w, a);
        }
        __builtin_amdgcn_wave_barrier();
    }"""

NEW_STRIP = """    // EXP_PD k steps of words in flight (registers), consumed in order: the same arithmetic as one step ahead
    constexpr int PD = EXP_PD;
    uint32_t pf[PD][LW];
    uint4 xa[PD];
#pragma unroll
    for (int u = 0; u < PD; ++u)
        if (u < n) {
            const uint32_t* p = tiles + (int64_t)(kt0 + u) * step;
#pragma unroll
            for (int i = 0; i < LW; ++i) pf[u][i] = __builtin_nontemporal_load(gl(p) + i * 32 + lane);
            xa[u] = *gl(reinterpret_cast<const uint4*>(xrow + 16 * (kt0 + u) + 8 * h));
        }
    for (int j0 = 0; j0 < n; j0 += PD) {
#pragma unroll
        for (int u = 0; u < PD; ++u) {
            const int j = j0 + u;
            if (j < n) {
#pragma unroll
                for (int i = 0; i < LW; ++i) st[i * 32 + lane] = pf[u][i];
                const uint4 x = xa[u];
                if (j + PD < n) {
                    const uint32_t* p = tiles + (int64_t)(kt0 + j + PD) * step;
#pragma unroll
                    for (int i = 0; i < LW; ++i) pf[u][i] = __builtin_nontemporal_load(gl(p) + i * 32 + lane);
                    xa[u] = *gl(reinterpret_cast<const uint4*>(xrow + 16 * (kt0 + j + PD) + 8 * h));
                }
                lds_fence();
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    uint32_t own[4];
                    decode_own<K2>(st + t * TW, win, own);
                    float a = dot2(own[0], x.x, s[t]);
                    a = dot2(own[1], x.y, a);
                    a = dot2(own[2], x.z, a);
                    s[t] = dot2(own[3], x.w, a);
                }
                __builtin_amdgcn_wave_barrier();
            }
        }
    }"""

LD128_STRIP = """    // a k step's words as 16-byte loads (lane l: words 4 l .. 4 l + 3 of each 128), the tail as 4-byte ones
    constexpr int TOT = 8 * TW, N4 = TOT / 128, NR = (TOT % 128) / 32;
    typedef uint32_t v4u __attribute__((ext_vector_type(4)));
    v4u pf4[N4 > 0 ? N4 : 1];
    uint32_t pfr[NR > 0 ? NR : 1];
    uint4 xa;
    {
        const uint32_t* p = tiles + (int64_t)kt0 * step;
#pragma unroll
        for (int i = 0; i < N4; ++i) pf4[i] = __builtin_nontemporal_load(gl(reinterpret_cast<const v4u*>(p + i * 128 + 4 * lane)));
#pragma unroll
        for (int i = 0; i < NR; ++i) pfr[i] = __builtin_nontemporal_load(gl(p) + N4 * 128 + i * 32 + lane);
        xa = *gl(reinterpret_cast<const uint4*>(xrow + 16 * kt0 + 8 * h));
    }
    for (int j = 0; j < n; ++j) {
#pragma unroll
        for (int i = 0; i < N4; ++i) *reinterpret_cast<v4u*>(st + i * 128 + 4 * lane) = pf4[i];
#pragma unroll
        for (int i = 0; i < NR; ++i) st[N4 * 128 + i * 32 + lane] = pfr[i];
        const uint4 x = xa;
        if (j + 1 < n) {
            const uint32_t* p = tiles + (int64_t)(kt0 + j + 1) * step;
#pragma unroll
            for (int i = 0; i < N4; ++i) pf4[i] = __builtin_nontemporal_load(gl(reinterpret_cast<const v4u*>(p + i * 128 + 4 * lane)));
#pragma unroll
            for (int i = 0; i < NR; ++i) pfr[i] = __builtin_nontemporal_load(gl(p) + N4 * 128 + i * 32 + lane);
            xa = *gl(reinterpret_cast<const uint4*>(xrow + 16 * (kt0 + j + 1) + 8 * h));
        }
        lds_fence();
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            uint32_t own[4];
            decode_own<K2>(st + t * TW, win, own);
            float a = dot2(own[0], x.x, s[t]);
            a = dot2(own[1], x.y, a);
            a = dot2(own[2], x.z, a);
            s[t] = dot2(own[3], x.w, a);
        }
        __builtin_amdgcn_wave_barrier();
    }"""

for exp in exps:
    if exp == "only1":
        rep("    constexpr int W = 4, RR = 8;", "    constexpr int W = 4, RR = 1;")
        rep("            for (int i = 0; i < 4; ++i) red[wave][2 * i + (lane >> 4)][t * 16 + (lane & 15)] = acc[t][4 * hp + i];",
            "            for (int i = 0; i < 4; ++i) if (2 * i + (lane >> 4) < RR) red[wave][2 * i + (lane >> 4)][t * 16 + (lane & 15)] = acc[t][4 * hp + i];")
        rep("    if (nrows == 1) strip_tiles_1row<K2_>", "    if (true) strip_tiles_1row<K2_>")
        rep("#if !defined(EXL3_WIDE) || EXL3_WIDE", "#if 0")
        rep("        if (R == 1) {\n", "        if (true) {\n")
        rep("        if (hp > 0 && rows_sh[8] < 0) break;", "        if (hp > 0) break;")
    elif exp == "pkhash":
        rep(OLD_PAIR, PK_PAIR)
    elif exp in ("pd2", "pd3", "pd4"):
        rep(OLD_STRIP, NEW_STRIP)
        s = "#define EXP_PD %s\n" % exp[2] + s
    elif exp == "ld128":
        rep(OLD_STRIP, LD128_STRIP)
    elif exp == "nodecode":
        # timing only (wrong numbers): the MUL1 decode replaced by one AND, everything else (loads, staging, LDS
        # reads, dot2, reductions) as it is: the kernels' ceiling without the decode's ALU
        rep(OLD_PAIR, """    const uint32_t s = u & 0x3bff3bffu;""")
        rep("""    h2 hv = __builtin_bit_cast(h2, s);  // bitcast-ok: a whole u32 to f16x2, no lane extract
    hv = __builtin_elementwise_fma(hv, __builtin_bit_cast(h2, 0x1eee1eeeu), __builtin_bit_cast(h2, 0xc931c931u));  // bitcast-ok: whole-vector constants
    return __builtin_bit_cast(uint32_t, hv);  // bitcast-ok: a whole f16x2 to u32, no lane extract""", """    return s;""")
    elif exp == "kil":
        # timing only: glinear's 1-row waves take interleaved k tiles (wave w: base + w + j WK) instead of a contiguous
        # range (another summation order: not row-invariant against the rows path; wide paths unchanged)
        rep("""                                                 uint32_t* st, int lane, f32x8 (&acc)[8]) {
    constexpr int TW = 4 * K2, LW = 8 * TW / 32;
    if (n <= 0) return;
    const LaneWin<K2> win(lane);
    const int h = lane >> 4;
    float s[8];""", """                                                 uint32_t* st, int lane, f32x8 (&acc)[8], int xs = 16) {
    constexpr int TW = 4 * K2, LW = 8 * TW / 32;
    if (n <= 0) return;
    const LaneWin<K2> win(lane);
    const int h = lane >> 4;
    float s[8];""")
        rep("""        xa = *gl(reinterpret_cast<const uint4*>(xrow + 16 * kt0 + 8 * h));
    }
    for (int j = 0; j < n; ++j) {""", """        xa = *gl(reinterpret_cast<const uint4*>(xrow + xs * kt0 + 8 * h));
    }
    for (int j = 0; j < n; ++j) {""")
        rep("""            xa = *gl(reinterpret_cast<const uint4*>(xrow + 16 * (kt0 + j + 1) + 8 * h));
        }
        lds_fence();
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            uint32_t own[4];
            decode_own<K2>(st + t * TW, win, own);
            float a = dot2(own[0], x.x, s[t]);""", """            xa = *gl(reinterpret_cast<const uint4*>(xrow + xs * (kt0 + j + 1) + 8 * h));
        }
        lds_fence();
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            uint32_t own[4];
            decode_own<K2>(st + t * TW, win, own);
            float a = dot2(own[0], x.x, s[t]);""")
        rep("""            strip_tiles_1row<K2>(tiles, L.stride_k, kt0, per_warp, xh + (size_t)m0 * L.ldx, stage[wave], lane, acc);""",
            """            {
                const int kb = split * (per_warp * WK) + wave;
                strip_tiles_1row<K2>(tiles + (int64_t)kb * L.stride_k, (int64_t)WK * L.stride_k, 0, per_warp,
                                     xh + (size_t)m0 * L.ldx + 16 * kb, stage[wave], lane, acc, 16 * WK);
            }""")
    elif exp == "mul24":
        s = "#define EXL3_MUL24 1\n" + s
    else:
        raise SystemExit("unknown experiment " + exp)
open(out, "w").write(s)
