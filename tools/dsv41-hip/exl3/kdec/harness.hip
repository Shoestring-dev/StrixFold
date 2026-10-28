// W-exl3 engine-ABI test of the kit's EXL3 code object (zig/kernels/hip/dsv41/exl3.hip): every kernel is loaded with
// hipModuleLoad and launched with hipModuleLaunchKernel, its arguments packed as the family's launchers pack them
// (exl3_linear.zig rotMany / glinear, exl3_prefill.zig rotIn / unpack, exl3_experts.zig prompt / decodeLaunches: the
// same plans, grids, blocks, dynamic LDS and scratch), on real LibertAI DeepSeek-V4.1-Flash-1017 2.9 bpw tensors in
// the layouts weights.zig builds (dense linears in strips; experts in one table set, the shared expert last).
//
// Modes (JSON lines on stdout):
//   glinear R GROUP    rot_many + glinear: w-ref contract per layer (fp32 y), bf16 y == bf16(fp32 y), the folded RoPE
//                      and next-layer rotation bit for bit against a CPU emulation of the kernels' fp32 order,
//                      counters zero after, cold-cache timing
//   prefill M LAYER    rot_in (bit for bit vs CPU emulation) and unpack (bit for bit vs the oracle's decode)
//   dexperts R LAYER   decode experts (decode_prep, grouped_cp gate/up, grouped_cp down): contract vs FP64, scratch
//                      protocol (epoch, ready, counters), timing
//   pexperts M LAYER   prompt experts (eight launches): contract on sampled rows, timing
//   pfexperts M LAYER  the AMD kit's prompt experts as exl3_experts.zig pfPlan launches them (a gu and a down launch per
//                      K2 of the table): contract on sampled rows, the grouped path on the same routes, timing
#include "hostcommon.h"
#include "abi.h"

static hipModule_t g_mod;
// LEAN=1: single-row launches take the single-row entry points (tf_ds_exl3_glinear1_*, *_grouped_cp1_*), as the
// launchers do when the kit has them
static const bool g_lean = getenv("LEAN") && atoi(getenv("LEAN")) != 0;
// SPLIT=1 (with LEAN=1): windows of 2+ rows take grouped_cp1 (one-member places) and grouped_cpm (the others) as two
// launches each for gate / up and down, as exl3_experts.zig decode does when the kit has both
static const bool g_split = getenv("SPLIT") && atoi(getenv("SPLIT")) != 0;

// HIP-graph timing (how the served engine replays a round): N launch groups (iterations 0 .. N-1, cold weights /
// routes as the caller rotates them) captured into one graph, replayed REPS times; mean per group (us)
static double time_graph(const std::function<void(int)>& launch, int n, int reps)
{
    for (int i = 0; i < 8; ++i) launch(i);                    // module loads and first-touch outside the capture
    HC(hipStreamSynchronize(g_stream));
    hipGraph_t g;
    hipGraphExec_t ge;
    HC(hipStreamBeginCapture(g_stream, hipStreamCaptureModeRelaxed));
    for (int i = 0; i < n; ++i) launch(i);
    HC(hipStreamEndCapture(g_stream, &g));
    HC(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    for (int i = 0; i < 2; ++i) HC(hipGraphLaunch(ge, g_stream));   // warm
    HC(hipStreamSynchronize(g_stream));
    hipEvent_t a, b;
    HC(hipEventCreate(&a)); HC(hipEventCreate(&b));
    HC(hipEventRecord(a, g_stream));
    for (int r = 0; r < reps; ++r) HC(hipGraphLaunch(ge, g_stream));
    HC(hipEventRecord(b, g_stream));
    HC(hipStreamSynchronize(g_stream));
    float ms;
    HC(hipEventElapsedTime(&ms, a, b));
    HC(hipEventDestroy(a)); HC(hipEventDestroy(b));
    HC(hipGraphExecDestroy(ge)); HC(hipGraphDestroy(g));
    return 1000.0 * ms / (reps * n);
}
static hipModule_t g_pfmod;   // the kit's exl3_pf.hsaco (PF_HSACO), loaded on first use

static hipFunction_t fn(const std::string& name)
{
    static std::map<std::string, hipFunction_t> cache;
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    hipFunction_t f;
    if (hipModuleGetFunction(&f, g_mod, name.c_str()) != hipSuccess)
    {
        if (!g_pfmod)
        {
            const char* pf = getenv("PF_HSACO");
            const std::string path = pf ? pf : std::string(getenv("HOME")) + "/tfp/exl3/tmp/kit/exl3_pf.hsaco";
            HC(hipModuleLoad(&g_pfmod, path.c_str()));
        }
        if (hipModuleGetFunction(&f, g_pfmod, name.c_str()) != hipSuccess) DIE("no kernel %s in the code objects", name.c_str());
    }
    cache[name] = f;
    return f;
}

template <typename P>
static void launch(hipFunction_t f, dim3 grid, dim3 block, unsigned smem, const P& p)
{
    size_t sz = sizeof(P);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, (void*) &p, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    HC(hipModuleLaunchKernel(f, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, g_stream, nullptr, cfg));
}

// ---- host emulation of the kernels' fp32 epilogue arithmetic (IEEE binary32, no contraction: -ffp-contract=off) ----

static const float HAD = 0.08838834764831845f;
static float bf_to_f(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f_to_bf(float f)
{
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
static uint16_t f_to_h(float f) { return exo_f64_to_f16((double) f); }
static float h_to_f(uint16_t h) { return (float) exo_f16_to_f64(h); }

// decode.cuh fwht128 on 128 values held as v[lane][j] = element 4 lane + j
static void emu_fwht(float (&v)[32][4])
{
    for (int l = 0; l < 32; ++l)
    {
        const float a = v[l][0] + v[l][1], b = v[l][0] - v[l][1], c = v[l][2] + v[l][3], d = v[l][2] - v[l][3];
        v[l][0] = a + c; v[l][1] = b + d; v[l][2] = a - c; v[l][3] = b - d;
    }
    for (int m = 1; m < 32; m <<= 1)
    {
        float w[32][4];
        for (int l = 0; l < 32; ++l)
            for (int j = 0; j < 4; ++j)
            {
                const float o = v[l ^ m][j];
                w[l][j] = (l & m) ? o - v[l][j] : v[l][j] + o;
            }
        memcpy(v, w, sizeof w);
    }
}

// rot128_in on 128 values x (fp32) with suh s (fp16 bits): the rotated fp16 bits
static void emu_rot128(const float* x, const uint16_t* s, uint16_t* out)
{
    float v[32][4];
    for (int l = 0; l < 32; ++l)
    {
        const float* xv = x + 4 * l;
        float sv[4];
        for (int j = 0; j < 4; ++j) sv[j] = h_to_f(s[4 * l + j]);
        const float p1 = xv[1] * sv[1], p3 = xv[3] * sv[3];
        const float a = fmaf(xv[0], sv[0], p1), b = fmaf(xv[0], sv[0], -p1);
        const float c = fmaf(xv[2], sv[2], p3), d = fmaf(xv[2], sv[2], -p3);
        v[l][0] = a + c; v[l][1] = b + d; v[l][2] = a - c; v[l][3] = b - d;
    }
    for (int m = 1; m < 32; m <<= 1)
    {
        float w[32][4];
        for (int l = 0; l < 32; ++l)
            for (int j = 0; j < 4; ++j)
            {
                const float o = v[l ^ m][j];
                w[l][j] = (l & m) ? o - v[l][j] : v[l][j] + o;
            }
        memcpy(v, w, sizeof w);
    }
    for (int l = 0; l < 32; ++l)
        for (int j = 0; j < 4; ++j) out[4 * l + j] = f_to_h(v[l][j] * HAD);
    if (getenv("DEBUG_ROT"))
        for (int l = 0; l < 32; ++l)
            for (int j = 0; j < 4; ++j)
            {
                const float f = v[l][j] * HAD;
                const double lo = h_to_f(out[4 * l + j]);
                if (fabs(fabs((double) f - lo) - fabs((double) h_to_f(out[4 * l + j] ^ 1) - lo) / 2) < 1e-12)
                    fprintf(stderr, "tie at %d: f32 %.10g (bits %08x) -> %04x\n", 4 * l + j, f, *(const uint32_t*) &f, out[4 * l + j]);
            }
}

// ---- dense layers in the engine's layout ------------------------------------------------------------------------------

struct Plan { int sk, wk; };
static Plan plan(int k, int n)
{
    // exl3_linear.zig plan(); PLAN_BLOCKS / PLAN_MIN_TILES override its two constants (tuning sweeps only)
    static const int blocks = env_int("PLAN_BLOCKS", 192), min_tiles = env_int("PLAN_MIN_TILES", 8);
    const int kt = k / 16, nb = n / 128;
    int sk = 1, wk = nb >= 64 ? 8 : 4;
    if (nb < 64) while (nb * sk < blocks && sk < 64) sk *= 2;
    while ((wk > 4 || sk > 1) && (kt % (sk * wk) != 0 || kt / (sk * wk) < min_tiles))
    {
        if (wk > 4) wk = 4;
        else if (sk > 1) sk /= 2;
        else break;
    }
    // PLAN_TABLE=1: exl3_linear.zig planHip, the gfx1151 table (fewer K splits where measured faster); PLAN_FORCE=
    // "K:N:SK,...": force SK for those shapes (sweeps)
    if (env_int("PLAN_TABLE", 0))
    {
        static const int tuned[][3] = {{5120, 1280, 4}, {5120, 512, 4}, {1280, 4096, 1}};
        for (auto& t : tuned) if (k == t[0] && n == t[1] && t[2] < sk) sk = t[2];
    }
    if (const char* f = getenv("PLAN_FORCE"))
        for (const char* q = f; *q;)
        {
            int fk = 0, fn = 0, fs = 0;
            if (sscanf(q, "%d:%d:%d", &fk, &fn, &fs) == 3 && fk == k && fn == n) sk = fs;
            while (*q && *q != ',') ++q;
            if (*q) ++q;
        }
    return {sk, wk};
}

// weights.zig strips: trellis int16 [kt, nt, w16] -> int32 words [nt / 8, kt, 8, w16 / 2]
static std::vector<uint16_t> to_strips(const Lin& l)
{
    const int kt = l.in / 16, nt = l.out / 16, w16 = 16 * l.K;
    std::vector<uint16_t> out(l.tr.size());
    for (int j = 0; j < nt / 8; ++j)
        for (int i = 0; i < kt; ++i)
            memcpy(&out[((size_t) j * kt + i) * 8 * w16], &l.tr[((size_t) i * nt + j * 8) * w16], (size_t) 8 * w16 * 2);
    return out;
}

struct ELin
{
    Lin h;
    int k = 0, n = 0, k2 = 0;
    const uint32_t* words = nullptr;
    const uint16_t* suh = nullptr;
    const uint16_t* svh = nullptr;
    int64_t stride_k() const { return 8 * 4 * k2; }
    int64_t stride_nb() const { return (int64_t) (k / 16) * 8 * 4 * k2; }
};

static ELin eupload(const Lin& l)
{
    ELin e;
    e.h = l;
    e.k = l.in; e.n = l.out; e.k2 = 2 * l.K;
    e.words = (const uint32_t*) dupload(to_strips(l));
    e.suh = dupload(l.suh);
    e.svh = dupload(l.svh);
    return e;
}

static Lin make_lin(const std::string& name, int slice, int a, int b)
{
    Lin l = load_lin(name);
    if (slice == 1) return slice_cols(l, a, b);
    if (slice == 2) return slice_rows(l, a, b);
    return l;
}

struct Call
{
    const ELin* L;
    const void* x; int64_t ldx; int x_dtype;
    uint16_t* xh;
    void* y; int64_t ldy; int y_dtype;
    int32_t* counters;
    const uint16_t* rsuh = nullptr; uint16_t* rxh = nullptr; int64_t ldr = 0; int roff = 0;
    const float* rcos = nullptr; const float* rsin = nullptr; const int64_t* rpos = nullptr; int rhd = 0, rrd = 0;
};

static void rot_many(const std::vector<Call>& calls, int m)
{
    tf_ds_exl3_rargs a;
    memset(&a, 0, sizeof a);
    a.n = (int) calls.size();
    a.M = m;
    int warps = 0;
    for (size_t i = 0; i < calls.size(); ++i)
    {
        const Call& c = calls[i];
        a.l[i] = {(uint64_t) c.x, (uint64_t) c.L->suh, (uint64_t) c.xh, c.ldx, c.x_dtype, c.L->k, warps, 0};
        warps += m * (c.L->k / 128);
    }
    static hipFunction_t f = fn("tf_ds_exl3_rot_many");
    launch(f, dim3((warps + 3) / 4), dim3(128), 0, a);
}

static size_t z_floats(const std::vector<Call>& calls, int m)
{
    size_t n = 0;
    for (auto& c : calls) { const Plan p = plan(c.L->k, c.L->n); if (p.sk > 1) n += (size_t) p.sk * m * c.L->n; }
    return n;
}

// exl3_linear.zig glinear: each (bits, warps) set in first-seen order, its split layers' partials in z
static int glinear(const std::vector<Call>& calls, int m, float* z)
{
    std::vector<bool> done(calls.size(), false);
    int launches = 0;
    for (size_t i = 0; i < calls.size(); ++i)
    {
        if (done[i]) continue;
        const Plan pf = plan(calls[i].L->k, calls[i].L->n);
        tf_ds_exl3_gargs a;
        memset(&a, 0, sizeof a);
        a.M = m;
        int blocks = 0;
        size_t zoff = 0;
        for (size_t j = i; j < calls.size(); ++j)
        {
            const Call& c = calls[j];
            const Plan pj = plan(c.L->k, c.L->n);
            if (done[j] || c.L->k2 != calls[i].L->k2 || pj.wk != pf.wk) continue;
            done[j] = true;
            tf_ds_exl3_glayer& g = a.l[a.n];
            g.xh = (uint64_t) c.xh; g.ldx = c.L->k; g.T = (uint64_t) c.L->words;
            g.stride_k = c.L->stride_k(); g.stride_nb = c.L->stride_nb(); g.svh = (uint64_t) c.L->svh;
            g.y = (uint64_t) c.y; g.y_dtype = c.y_dtype; g.ldy = c.ldy; g.K = c.L->k; g.N = c.L->n; g.SK = pj.sk;
            g.counters = (uint64_t) c.counters; g.first = blocks;
            if (c.rsuh) { g.rsuh = (uint64_t) c.rsuh; g.rxh = (uint64_t) c.rxh; g.ldr = c.ldr; g.roff = c.roff; }
            if (c.rcos) { g.rcos = (uint64_t) c.rcos; g.rsin = (uint64_t) c.rsin; g.rpos = (uint64_t) c.rpos; g.rhd = c.rhd; g.rrd = c.rrd; }
            if (pj.sk > 1) { g.Z = (uint64_t) (z + zoff); zoff += (size_t) pj.sk * m * c.L->n; }
            blocks += (c.L->n / 128) * pj.sk;
            a.n++;
        }
        const std::string name = std::string(g_lean && m == 1 ? "tf_ds_exl3_glinear1_k" : "tf_ds_exl3_glinear_k") + std::to_string(calls[i].L->k2) + "_w" + std::to_string(pf.wk);
        launch(fn(name), dim3(blocks), dim3(pf.wk * 32), (unsigned) (pf.wk * std::min(m, 8) * 128 * 4), a);
        launches++;
    }
    return launches;
}

struct GroupCase { std::string name; std::vector<std::string> tensors; int slice, a, b; };

static std::vector<GroupCase> groups()
{
    return {
        {"qa_kv", {"layers.20.attn.wq_a", "layers.20.attn.wkv"}, 0, 0, 0},
        {"qb", {"layers.20.attn.wq_b"}, 1, 0, 16384},
        {"woa", {"layers.39.attn.wo_a.slice.0", "layers.39.attn.wo_a.slice.1", "layers.39.attn.wo_a.slice.2", "layers.39.attn.wo_a.slice.3"}, 0, 0, 0},
        {"wob", {"layers.39.attn.wo_b"}, 2, 0, 4096},
        {"idx", {"layers.20.attn.indexer.wq_b", "layers.20.attn.indexer.wk", "layers.20.attn.compressor.wkv"}, 0, 0, 0},
        {"shared", {"layers.20.ffn.shared_experts.w1", "layers.39.ffn.shared_experts.w3"}, 0, 0, 0},
        {"shared_w2", {"layers.39.ffn.shared_experts.w2"}, 0, 0, 0},
        {"head", {"head"}, 1, 0, 64640},
        // k-dec2 (M1.exl3-numeric receipt): the other (K2, warps) sets the engine reaches; not in "all" (timing
        // comparisons keep the 8 groups above); "receipt" = all 14 dense groups the engine launches
        {"qa_kv0", {"layers.0.attn.wq_a", "layers.0.attn.wkv"}, 0, 0, 0},
        {"eng1", {"layers.1.engram.wkv"}, 2, 0, 3072},
        {"eng14", {"layers.14.engram.wkv"}, 2, 0, 3072},
        {"mtp_qa", {"mtp.0.attn.wq_a", "mtp.0.attn.wkv"}, 0, 0, 0},
        {"mtp_qb", {"mtp.0.attn.wq_b"}, 1, 0, 16384},
        {"main_proj", {"mtp.0.main_proj"}, 0, 0, 0},
    };
}

// gname: "all" (the first 8 groups), "receipt" (every group but the harness-only shared / shared_w2), or a name list
static bool group_selected(const std::string& gname, const std::string& g)
{
    static const char* extra[] = {"qa_kv0", "eng1", "eng14", "mtp_qa", "mtp_qb", "main_proj"};
    bool is_extra = false;
    for (const char* e : extra) is_extra |= g == e;
    if (gname == "all") return !is_extra;
    if (gname == "receipt") return g != "shared" && g != "shared_w2";
    const std::string l = "," + gname + ",";
    return l.find("," + g + ",") != std::string::npos;
}

// k-dec2 MALL=1: plain 16-byte loads over a range (what a prefetch leaves in gfx1151's memory-side cache), and a cold
// flush (256 MB of other reads)
typedef uint32_t mall_v4u __attribute__((ext_vector_type(4)));
__global__ void __launch_bounds__(256) mall_touch(const mall_v4u* __restrict__ p, size_t n16, uint32_t* out)
{
    uint32_t acc = 0;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) { const mall_v4u v = p[i]; acc += v.x ^ v.w; }
    if (acc == 0x12345678u) out[0] = acc;
}

static int mode_glinear(int R, const std::string& gname)
{
    int rc = 0;
    for (const GroupCase& gc : groups())
    {
        if (!group_selected(gname, gc.name)) continue;
        std::vector<Lin> L;
        for (auto& t : gc.tensors) L.push_back(make_lin(t, gc.slice, gc.a, gc.b));
        size_t bytes = 0;
        for (auto& l : L) bytes += l.tr_bytes();
        const size_t target = (size_t) env_int("ROT_MB", 384) << 20;
        const int C = (int) std::min<size_t>(64, std::max<size_t>(2, (target + bytes - 1) / bytes));
        std::vector<std::vector<ELin>> copies(C);
        for (int c = 0; c < C; ++c) for (auto& l : L) copies[c].push_back(eupload(l));
        const int n = (int) L.size();

        // inputs: bf16 rows (the family's residual-stream dtype), each layer its own
        std::vector<std::vector<uint16_t>> xb(n);
        std::vector<uint16_t*> dx(n), dxh(n);
        std::vector<float*> dy(n);
        std::vector<uint16_t*> dyb(n);
        std::vector<int32_t*> cnt(n);
        for (int i = 0; i < n; ++i)
        {
            std::vector<uint16_t> xh16 = rand_x((size_t) R * L[i].in);
            xb[i].resize(xh16.size());
            for (size_t j = 0; j < xh16.size(); ++j) xb[i][j] = f_to_bf((float) exo_f16_to_f64(xh16[j]));
            dx[i] = dupload(xb[i]);
            dxh[i] = dalloc<uint16_t>((size_t) R * L[i].in);
            dy[i] = dalloc<float>((size_t) R * L[i].out);
            dyb[i] = dalloc<uint16_t>((size_t) R * L[i].out);
            cnt[i] = dalloc<int32_t>((size_t) 8 * L[i].out / 128);
            dzero(cnt[i], (size_t) 8 * L[i].out / 128 * 4);
        }
        auto make_calls = [&](int c, bool bf16_out) {
            std::vector<Call> calls;
            for (int i = 0; i < n; ++i)
                calls.push_back({&copies[c][i], dx[i], L[i].in, 1, dxh[i], bf16_out ? (void*) dyb[i] : (void*) dy[i], L[i].out, bf16_out ? 1 : 2, cnt[i]});
            return calls;
        };
        std::vector<Call> calls0 = make_calls(0, false);
        float* dz = dalloc<float>(std::max<size_t>(1, z_floats(calls0, R)));

        // correctness: fp32 outputs against w-ref's contract
        rot_many(calls0, R);
        const int launches = 1 + glinear(calls0, R, dz);
        HC(hipStreamSynchronize(g_stream));
        Contract ct;
        ct.pass = true;
        std::vector<std::vector<float>> yf(n);
        for (int i = 0; i < n; ++i)
        {
            yf[i] = download(dy[i], (size_t) R * L[i].out);
            std::vector<double> xd((size_t) R * L[i].in);
            for (size_t j = 0; j < xd.size(); ++j) xd[j] = bf_to_f(xb[i][j]);
            std::vector<uint16_t> wq = cpu_decode(L[i]);
            std::vector<double> wy, wB, wP;
            wref_ref(L[i], R, xd, wq, wy, wB, wP);
            Contract ci = grade(yf[i], wy, wB, wP, R, L[i].out);
            ct.bound_ratio = std::max(ct.bound_ratio, ci.bound_ratio); ct.prob_exceed += ci.prob_exceed;
            ct.nrmse = std::max(ct.nrmse, ci.nrmse); ct.row_nrmse_max = std::max(ct.row_nrmse_max, ci.row_nrmse_max);
            ct.nonfinite += ci.nonfinite; ct.n += ci.n; ct.pass = ct.pass && ci.pass;
        }
        // counters left zero
        size_t nonzero = 0;
        for (int i = 0; i < n; ++i) { auto cv = download(cnt[i], (size_t) 8 * L[i].out / 128); for (int v : cv) nonzero += v != 0; }
        // bf16 outputs: the same sums rounded once
        std::vector<Call> callsb = make_calls(0, true);
        rot_many(callsb, R);
        glinear(callsb, R, dz);
        size_t bf_diff = 0;
        for (int i = 0; i < n; ++i)
        {
            auto yb = download(dyb[i], (size_t) R * L[i].out);
            for (size_t j = 0; j < yb.size(); ++j) bf_diff += yb[j] != f_to_bf(yf[i][j]);
        }
        // the folded next-layer rotation (layer 0, bf16 y) and RoPE (layer 0 when its width holds whole 512-column heads)
        size_t rot_diff = 0, rope_diff = 0;
        {
            const int N0 = L[0].out;
            std::vector<uint16_t> rsuh(N0);
            for (int j = 0; j < N0; ++j) rsuh[j] = (g_rng() & 1) ? 0xbc00 : 0x3c00;   // +-1
            uint16_t* drsuh = dupload(rsuh);
            uint16_t* drxh = dalloc<uint16_t>((size_t) R * N0);
            std::vector<Call> cr = make_calls(0, true);
            cr[0].rsuh = drsuh; cr[0].rxh = drxh; cr[0].ldr = N0; cr[0].roff = 0;
            rot_many(cr, R);
            glinear(cr, R, dz);
            auto yb = download(dyb[0], (size_t) R * N0);
            auto rx = download(drxh, (size_t) R * N0);
            for (int r = 0; r < R; ++r)
                for (int b = 0; b < N0 / 128; ++b)
                {
                    float u[128];
                    uint16_t e[128];
                    for (int j = 0; j < 128; ++j) u[j] = bf_to_f(yb[(size_t) r * N0 + b * 128 + j]);
                    emu_rot128(u, &rsuh[b * 128], e);
                    for (int j = 0; j < 128; ++j)
                    {
                        const uint16_t gv = rx[(size_t) r * N0 + b * 128 + j];
                        if (e[j] != gv && rot_diff < 8 && getenv("DEBUG_ROT"))
                        {
                            fprintf(stderr, "rot_out r %d col %d gpu %04x (%.9g) emu %04x (%.9g)\n", r, b * 128 + j, gv, h_to_f(gv), e[j], h_to_f(e[j]));
                            fprintf(stderr, "dump u");
                            for (int q = 0; q < 128; ++q) fprintf(stderr, " %04x", yb[(size_t) r * N0 + b * 128 + q]);
                            fprintf(stderr, "\ndump s");
                            for (int q = 0; q < 128; ++q) fprintf(stderr, " %04x", rsuh[b * 128 + q]);
                            fprintf(stderr, "\ndump g");
                            for (int q = 0; q < 128; ++q) fprintf(stderr, " %04x", rx[(size_t) r * N0 + b * 128 + q]);
                            fprintf(stderr, "\n");
                        }
                        rot_diff += e[j] != gv;
                    }
                }
            if (N0 % 512 == 0)
            {
                const int rhd = 512, rrd = 64, maxpos = 4096;
                std::vector<float> cs((size_t) maxpos * rrd / 2), sn(cs.size());
                for (size_t j = 0; j < cs.size(); ++j) { const double t = 0.001 * (double) j; cs[j] = (float) cos(t); sn[j] = (float) sin(t); }
                std::vector<int64_t> pos(R);
                for (int r = 0; r < R; ++r) pos[r] = (int64_t) (g_rng() % maxpos);
                float* dcs = dupload(cs);
                float* dsn = dupload(sn);
                int64_t* dpos = dupload(pos);
                std::vector<Call> cp = make_calls(0, true);
                cp[0].rcos = dcs; cp[0].rsin = dsn; cp[0].rpos = dpos; cp[0].rhd = rhd; cp[0].rrd = rrd;
                rot_many(cp, R);
                glinear(cp, R, dz);
                auto yr = download(dyb[0], (size_t) R * N0);
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < N0; c += 2)
                    {
                        const int d = c % rhd, lo = rhd - rrd;
                        float v0 = yf[0][(size_t) r * N0 + c], v1 = yf[0][(size_t) r * N0 + c + 1];
                        if (d >= lo)
                        {
                            const int j = (d - lo) >> 1;
                            const float cv = cs[(size_t) pos[r] * (rrd / 2) + j], sv = sn[(size_t) pos[r] * (rrd / 2) + j];
                            const float xe = bf_to_f(f_to_bf(v0)), xo = bf_to_f(f_to_bf(v1));
                            v0 = fmaf(xe, cv, -(xo * sv));
                            v1 = fmaf(xe, sv, xo * cv);
                        }
                        rope_diff += f_to_bf(v0) != yr[(size_t) r * N0 + c];
                        rope_diff += f_to_bf(v1) != yr[(size_t) r * N0 + c + 1];
                    }
            }
            else rope_diff = (size_t) -1;
        }
        HC(hipStreamSynchronize(g_stream));

        // cold-cache timing: rotating weight copies, fp32 outputs, the launches the family makes for the group
        const int iters = env_int("ITERS", 100), warm = 10;
        std::vector<std::vector<Call>> cc(C);
        for (int c = 0; c < C; ++c) cc[c] = make_calls(c, false);
        auto run = [&](int it) { rot_many(cc[it % C], R); glinear(cc[it % C], R, dz); };
        Stats st = time_events(run, warm, iters);
        const double b2b = time_b2b(run, warm, iters);
        const double gus = time_graph(run, std::min(C, 32), env_int("GRAPH_REPS", 20));
        // MALL=1: the launch pair (rot_many + glinear, event-timed) right after its copy's weights were touched into the
        // memory-side cache, against the same after a 256 MB flush (cold)
        double mall_hot = 0, mall_cold = 0;
        if (env_int("MALL", 0))
        {
            static char* flushbuf = nullptr;
            if (!flushbuf) HC(hipMalloc(&flushbuf, (size_t) 256 << 20));
            uint32_t* sink = dalloc<uint32_t>(4);
            auto touch = [&](int c) {
                for (int i = 0; i < n; ++i)
                    mall_touch<<<160, 256, 0, g_stream>>>((const mall_v4u*) copies[c][i].words, L[i].tr_bytes() / 16, sink);
            };
            auto flush = [&]() { mall_touch<<<640, 256, 0, g_stream>>>((const mall_v4u*) flushbuf, ((size_t) 256 << 20) / 16, sink); };
            for (int hot = 0; hot < 2; ++hot)
            {
                std::vector<float> ms;
                hipEvent_t a, b;
                HC(hipEventCreate(&a)); HC(hipEventCreate(&b));
                for (int it = 0; it < 4 + env_int("MALL_ITERS", 30); ++it)
                {
                    flush();
                    if (hot) touch(it % C);
                    HC(hipEventRecord(a, g_stream));
                    run(it);
                    HC(hipEventRecord(b, g_stream));
                    HC(hipEventSynchronize(b));
                    float t; HC(hipEventElapsedTime(&t, a, b));
                    if (it >= 4) ms.push_back(t * 1e3f);
                }
                std::sort(ms.begin(), ms.end());
                (hot ? mall_hot : mall_cold) = ms[ms.size() / 2];
                HC(hipEventDestroy(a)); HC(hipEventDestroy(b));
            }
        }
        std::string shapes;
        for (int i = 0; i < n; ++i)
        {
            const Plan p = plan(L[i].in, L[i].out);
            char buf[160];
            snprintf(buf, sizeof buf, "%s{\"tensor\":\"%s\",\"K\":%d,\"k\":%d,\"n\":%d,\"sk\":%d,\"wk\":%d}", i ? "," : "", L[i].name.c_str(), L[i].K, L[i].in, L[i].out, p.sk, p.wk);
            shapes += buf;
        }
        const bool ok = ct.pass && nonzero == 0 && bf_diff == 0 && rot_diff == 0 && (rope_diff == 0 || rope_diff == (size_t) -1);
        printf("{\"mode\":\"glinear\",\"group\":\"%s\",\"rows\":%d,\"layers\":[%s],\"launches\":%d,\"trellis_bytes\":%zu,\"copies\":%d,"
               "\"median_us\":%.2f,\"b2b_us\":%.2f,\"graph_us\":%.2f,\"gbs_median\":%.1f,\"gbs_b2b\":%.1f,\"sclk_min\":%d,\"sclk_med\":%d,\"load1\":%.2f,"
               "\"mall_cold_us\":%.2f,\"mall_hot_us\":%.2f,\"counters_nonzero\":%zu,\"bf16_y_diff\":%zu,\"rot_out_diff\":%zu,\"rope_diff\":%lld,",
               gc.name.c_str(), R, shapes.c_str(), launches, bytes, C, st.median_us, b2b, gus, bytes / (st.median_us * 1e3), bytes / (b2b * 1e3),
               g_clk_min, g_clk_med, g_load1, mall_cold, mall_hot, nonzero, bf_diff, rot_diff, rope_diff == (size_t) -1 ? -1LL : (long long) rope_diff);
        print_contract(ct);
        printf(",\"pass\":%s}\n", ok ? "true" : "false");
        fflush(stdout);
        rc |= ok ? 0 : 1;
        for (void* p : g_allocs) HC(hipFree(p));
        g_allocs.clear();
    }
    return rc;
}

// ---- prompt pieces: rot_in and unpack ---------------------------------------------------------------------------------

struct RotInP { uint64_t x; int32_t x_dtype; int64_t ldx; uint64_t suh, xh; int32_t k; };
struct UnpackP { uint64_t words, w; int32_t n; int64_t stride_k, stride_nb; };

static int mode_prefill(int M, const std::string& tensor)
{
    Lin l = load_lin(tensor);
    ELin e = eupload(l);
    std::vector<uint16_t> xh16 = rand_x((size_t) M * l.in), xb(xh16.size());
    for (size_t j = 0; j < xb.size(); ++j) xb[j] = f_to_bf((float) exo_f16_to_f64(xh16[j]));
    uint16_t* dx = dupload(xb);
    uint16_t* dxh = dalloc<uint16_t>(xb.size());
    uint16_t* dw = dalloc<uint16_t>((size_t) l.in * l.out);
    const RotInP rp{(uint64_t) dx, 1, l.in, (uint64_t) e.suh, (uint64_t) dxh, l.in};
    launch(fn("tf_ds_exl3_rot_in"), dim3((l.in / 128 + 3) / 4, M), dim3(128), 0, rp);
    const UnpackP up{(uint64_t) e.words, (uint64_t) dw, l.out, e.stride_k(), e.stride_nb()};
    hipFunction_t fu = fn("tf_ds_exl3_unpack_k" + std::to_string(e.k2));
    launch(fu, dim3(l.out / 16, l.in / 16), dim3(32), 0, up);
    HC(hipStreamSynchronize(g_stream));
    auto gxh = download(dxh, xb.size());
    auto gw = download(dw, (size_t) l.in * l.out);
    size_t rot_diff = 0, w_diff = 0;
    for (int r = 0; r < M; ++r)
        for (int b = 0; b < l.in / 128; ++b)
        {
            float u[128];
            uint16_t o[128];
            for (int j = 0; j < 128; ++j) u[j] = bf_to_f(xb[(size_t) r * l.in + b * 128 + j]);
            emu_rot128(u, &l.suh[b * 128], o);
            for (int j = 0; j < 128; ++j) rot_diff += o[j] != gxh[(size_t) r * l.in + b * 128 + j];
        }
    std::vector<uint16_t> wq = cpu_decode(l);
    for (size_t j = 0; j < wq.size(); ++j) w_diff += wq[j] != gw[j];
    const int iters = env_int("ITERS", 50);
    const double us_rot = time_b2b([&](int) { launch(fn("tf_ds_exl3_rot_in"), dim3((l.in / 128 + 3) / 4, M), dim3(128), 0, rp); }, 3, iters);
    const double us_unp = time_b2b([&](int) { launch(fu, dim3(l.out / 16, l.in / 16), dim3(32), 0, up); }, 3, iters);
    const bool ok = rot_diff == 0 && w_diff == 0;
    printf("{\"mode\":\"prefill\",\"tensor\":\"%s\",\"K\":%d,\"k\":%d,\"n\":%d,\"rows\":%d,\"rot_in_diff\":%zu,\"unpack_diff\":%zu,"
           "\"rot_in_us\":%.1f,\"unpack_us\":%.1f,\"unpack_out_gbs\":%.1f,\"load1\":%.2f,\"pass\":%s}\n",
           tensor.c_str(), l.K, l.in, l.out, M, rot_diff, w_diff, us_rot, us_unp, (double) l.in * l.out * 2 / (us_unp * 1e3), load1(), ok ? "true" : "false");
    fflush(stdout);
    return ok ? 0 : 1;
}

// ---- routed experts ------------------------------------------------------------------------------------------------------

// weights.zig Experts: routed experts 0..E-2 and the shared expert last; trellises in the checkpoint layout
struct EExperts
{
    int E = 0, D = 0, I = 0;
    std::vector<Lin> g, u, d;
    int64_t *gate_ptr, *up_ptr, *down_ptr;
    int32_t *gate_k2, *up_k2, *down_k2;
    uint16_t *suh_g, *suh_u, *svh_g, *svh_u, *suh_d, *svh_d;
    int k2_gu[2] = {99, 0}, k2_d[2] = {99, 0};
    size_t bytes_of(int e) const { return g[e].tr_bytes() + u[e].tr_bytes() + d[e].tr_bytes(); }
};

static EExperts load_eexperts(int layer, int routed)
{
    EExperts x;
    x.E = routed + 1;
    const int I = 1152;
    for (int e = 0; e < x.E; ++e)
    {
        // MTP=1: the drafter blocks' sets (mtp.N: 128 routed K4 experts, a K6 shared expert)
        const std::string base = getenv("MTP") ? "mtp." : "layers.";
        const std::string p = base + std::to_string(layer) + ".ffn." + (e < routed ? "experts." + std::to_string(e) : std::string("shared_experts")) + ".";
        x.g.push_back(slice_cols(load_lin(p + "w1"), 0, I));
        x.u.push_back(slice_cols(load_lin(p + "w3"), 0, I));
        x.d.push_back(slice_rows(load_lin(p + "w2"), 0, I));
    }
    x.D = x.g[0].in;
    x.I = I;
    std::vector<int64_t> gp, upp, dp;
    std::vector<int32_t> gk, uk, dk;
    std::vector<uint16_t> sg, su, vg, vu, sd, vd;
    for (int e = 0; e < x.E; ++e)
    {
        gp.push_back((int64_t) dupload(x.g[e].tr)); upp.push_back((int64_t) dupload(x.u[e].tr)); dp.push_back((int64_t) dupload(x.d[e].tr));
        gk.push_back(2 * x.g[e].K); uk.push_back(2 * x.u[e].K); dk.push_back(2 * x.d[e].K);
        sg.insert(sg.end(), x.g[e].suh.begin(), x.g[e].suh.end()); su.insert(su.end(), x.u[e].suh.begin(), x.u[e].suh.end());
        vg.insert(vg.end(), x.g[e].svh.begin(), x.g[e].svh.end()); vu.insert(vu.end(), x.u[e].svh.begin(), x.u[e].svh.end());
        sd.insert(sd.end(), x.d[e].suh.begin(), x.d[e].suh.end()); vd.insert(vd.end(), x.d[e].svh.begin(), x.d[e].svh.end());
        for (int k2 : {gk.back(), uk.back()}) { x.k2_gu[0] = std::min(x.k2_gu[0], k2); x.k2_gu[1] = std::max(x.k2_gu[1], k2); }
        x.k2_d[0] = std::min(x.k2_d[0], dk.back()); x.k2_d[1] = std::max(x.k2_d[1], dk.back());
    }
    x.gate_ptr = dupload(gp); x.up_ptr = dupload(upp); x.down_ptr = dupload(dp);
    x.gate_k2 = dupload(gk); x.up_k2 = dupload(uk); x.down_k2 = dupload(dk);
    x.suh_g = dupload(sg); x.suh_u = dupload(su); x.svh_g = dupload(vg); x.svh_u = dupload(vu); x.suh_d = dupload(sd); x.svh_d = dupload(vd);
    return x;
}

static std::string range_name(int lo, int hi)
{
    if (lo == 8 && hi == 8) return "lo8_hi8";
    if (lo >= 2 && hi <= 10) return "lo2_hi10";
    return "lo2_hi16";
}

// routes: R rows x slots, slots - 1 distinct routed experts a row (row r of iteration `it` from a fixed permutation) and
// the shared expert last; wts: routed in (0.05, 0.3), shared 1
static void make_routes(int R, int slots, int routed, int it, std::vector<int32_t>& pick, std::vector<float>& wts)
{
    pick.assign((size_t) R * slots, 0);
    wts.assign(pick.size(), 1.0f);
    std::mt19937_64 rng(1234567u + (uint64_t) it);
    for (int r = 0; r < R; ++r)
    {
        std::vector<int> perm(routed);
        std::iota(perm.begin(), perm.end(), 0);
        for (int s = 0; s < slots - 1; ++s)
        {
            const int j = s + (int) (rng() % (uint64_t) (routed - s));
            std::swap(perm[s], perm[j]);
            pick[(size_t) r * slots + s] = perm[s];
            wts[(size_t) r * slots + s] = 0.05f + 0.25f * (float) ((rng() >> 11) * 0x1p-53);
        }
        pick[(size_t) r * slots + slots - 1] = routed;
    }
}

// FP64 reference of rows `rows` (engine semantics: SwiGLU with g = min(g, L), u = clamp(u, +-L), fp32-free, the slot's
// weight after down; out_r = sum over slots): y [rows, D]
static std::vector<double> ref_experts(const EExperts& x, const std::vector<uint16_t>& xb, const std::vector<int32_t>& pick,
                                       const std::vector<float>& wts, int slots, const std::vector<int>& rows, float limit)
{
    const int D = x.D, I = x.I;
    std::vector<double> out((size_t) rows.size() * D, 0.0);
    std::map<int, std::array<std::vector<uint16_t>, 3>> wq;
    QuietScope q(LOCK_SH);
    for (size_t ri = 0; ri < rows.size(); ++ri)
    {
        const int r = rows[ri];
        std::vector<double> xr(D);
        for (int k = 0; k < D; ++k) xr[k] = bf_to_f(xb[(size_t) r * D + k]);
        for (int s = 0; s < slots; ++s)
        {
            const int e = pick[(size_t) r * slots + s];
            if (e < 0 || e >= x.E) continue;
            if (!wq.count(e)) wq[e] = {cpu_decode(x.g[e]), cpu_decode(x.u[e]), cpu_decode(x.d[e])};
            const auto& w = wq[e];
            std::vector<double> gy(I), gb(I), uy(I), ub(I), av(I), dy(D), db(D);
            exo_ref_gemv(1, D, I, xr.data(), x.g[e].suh.data(), x.g[e].svh.data(), w[0].data(), gy.data(), gb.data(), nullptr);
            exo_ref_gemv(1, D, I, xr.data(), x.u[e].suh.data(), x.u[e].svh.data(), w[1].data(), uy.data(), ub.data(), nullptr);
            for (int i = 0; i < I; ++i)
            {
                const double gg = std::min(gy[i], (double) limit), uu = std::min(std::max(uy[i], -(double) limit), (double) limit);
                av[i] = gg / (1.0 + exp(-gg)) * uu;
            }
            exo_ref_gemv(1, I, D, av.data(), x.d[e].suh.data(), x.d[e].svh.data(), w[2].data(), dy.data(), db.data(), nullptr);
            const double wt = wts[(size_t) r * slots + s];
            for (int j = 0; j < D; ++j) out[ri * D + j] += wt * dy[j];
        }
    }
    return out;
}

// numerics.json exl3.experts_grouped: NRMSE <= 2e-3 over the graded rows, the worst row's NRMSE <= 4x that
struct XGrade { double nrmse = 0, worst = 0; size_t nonfinite = 0; bool pass = false; };
static XGrade xgrade(const std::vector<float>& got, const std::vector<double>& ref, int nrows, int D)
{
    XGrade g;
    double se = 0, sr = 0;
    for (int r = 0; r < nrows; ++r)
    {
        double re = 0, rr = 0;
        for (int j = 0; j < D; ++j)
        {
            const size_t i = (size_t) r * D + j;
            if (!std::isfinite(got[i])) { g.nonfinite++; continue; }
            const double d = got[i] - ref[i];
            re += d * d; rr += ref[i] * ref[i];
        }
        se += re; sr += rr;
        g.worst = std::max(g.worst, sqrt(re / std::max(rr, 1e-300)));
    }
    g.nrmse = sqrt(se / std::max(sr, 1e-300));
    g.pass = g.nonfinite == 0 && g.nrmse <= 2e-3 && g.worst <= 8e-3;
    return g;
}

// exl3_experts.zig config(k, n, gateup): (n tiles, warps, K splits, tiles in flight)
static std::array<int, 4> xconfig(int k, int n, bool gateup)
{
    const std::array<int, 4> cands[] = {gateup ? std::array<int, 4>{8, 4, 4, 1} : std::array<int, 4>{8, 4, 1, 1}, {8, 4, 2, 1}, {8, 4, 1, 1}, {4, 4, 2, 2}, {4, 4, 1, 2}};
    for (auto& c : cands)
        if (k % (16 * c[2] * c[1]) == 0 && n % (16 * c[0]) == 0)
        {
            // SK_GU: gate / up's K splits (a plan experiment: the numerics change, every row alike)
            std::array<int, 4> r = c;
            if (gateup && getenv("SK_GU")) r[2] = env_int("SK_GU", c[2]);
            return r;
        }
    DIE("no tile setting for %d x %d", k, n);
}

struct PrepP {
    uint64_t x; int32_t x_stride; uint64_t pick, suh_g, suh_u, xg, xu; int32_t d, slots, e, r; uint64_t ids, count, members;
    int32_t maxm; uint64_t wts, y, add, out; int32_t d_out, store_y; uint64_t epoch;
};
struct CpP { uint64_t xa, xb, wa, wb, ka, kb, ids, count, members, z; int32_t k, n, p, splits, maxm, slots; tf_ds_exl3_decode_epi epi; };
// FUSED=1 (an experiment object with tf_ds_exl3_experts_grouped_cpf_*): gate / up and down as one launch
struct CpLaunchP { uint64_t xa, xb, wa, wb, ka, kb, ids, count, members, z; int32_t k, n, p, splits, maxm, slots, gx, gy, gz, pad; tf_ds_exl3_decode_epi epi; };
struct CpFusedP { CpLaunchP gu, dn; };
static_assert(sizeof(CpLaunchP) == 264 && offsetof(CpLaunchP, epi) == 120, "CpLaunch packing");
static CpLaunchP cp_launch(const CpP& c, dim3 g)
{
    CpLaunchP l;
    memcpy(&l, &c, 104);
    l.gx = (int32_t) g.x; l.gy = (int32_t) g.y; l.gz = (int32_t) g.z; l.pad = 0;
    l.epi = c.epi;
    return l;
}
static const bool g_fused = getenv("FUSED") && atoi(getenv("FUSED")) != 0;
static_assert(offsetof(CpP, epi) == 104 && sizeof(CpP) == 248, "GroupedCp packing");
static_assert(offsetof(PrepP, epoch) == 144, "DecodePrep packing");

struct DecodeScratch
{
    int rows, slots, maxu;
    uint16_t *xg, *xu, *xd;
    float *z, *y;
    int32_t *cnt_gu, *cnt_d, *epoch, *ready, *ready_cnt, *ids, *count, *members;
};

static DecodeScratch alloc_decode(int rows, int slots, int D, int I, int E)
{
    DecodeScratch s;
    s.rows = rows; s.slots = slots;
    const int p = rows * slots;
    s.maxu = std::min(p, E);
    const int zrow = std::max(2 * xconfig(D, I, true)[2] * I, xconfig(I, D, false)[2] * D);
    s.xg = dalloc<uint16_t>((size_t) p * D); dzero(s.xg, (size_t) p * D * 2);
    s.xu = dalloc<uint16_t>((size_t) p * D); dzero(s.xu, (size_t) p * D * 2);
    s.xd = dalloc<uint16_t>((size_t) p * I); dzero(s.xd, (size_t) p * I * 2);
    s.z = dalloc<float>((size_t) zrow * p); dzero(s.z, (size_t) zrow * p * 4);
    s.y = dalloc<float>((size_t) p * D); dzero(s.y, (size_t) p * D * 4);
    s.cnt_gu = dalloc<int32_t>((size_t) s.maxu * std::max(1, I / 128)); dzero(s.cnt_gu, (size_t) s.maxu * std::max(1, I / 128) * 4);
    s.cnt_d = dalloc<int32_t>((size_t) rows * std::max(1, D / 128)); dzero(s.cnt_d, (size_t) rows * std::max(1, D / 128) * 4);
    s.epoch = dalloc<int32_t>(1); dzero(s.epoch, 4);
    s.ready = dalloc<int32_t>(std::max(1, s.maxu)); dzero(s.ready, std::max(1, s.maxu) * 4);
    s.ready_cnt = dalloc<int32_t>(std::max(1, s.maxu)); dzero(s.ready_cnt, std::max(1, s.maxu) * 4);
    s.ids = dalloc<int32_t>(s.maxu); dzero(s.ids, s.maxu * 4);
    s.count = dalloc<int32_t>(1); dzero(s.count, 4);
    const size_t mb = (size_t) s.maxu * ((rows + 127) / 128 * 128);
    s.members = dalloc<int32_t>(mb); dfill(s.members, 0xff, mb * 4);
    return s;
}

// exl3_experts.zig decodeLaunches + decode
static void decode_experts(const EExperts& x, const DecodeScratch& sc, const uint16_t* dx, const int32_t* dpick, const float* dwts,
                           float* dout, int r, float limit, int upto = 3)
{
    const int D = x.D, I = x.I, E = x.E, slots = sc.slots, p = r * slots, maxu = std::min(p, E), mt = (r + 15) / 16;
    const auto cgu = xconfig(D, I, true);
    PrepP pp{(uint64_t) dx, D, (uint64_t) dpick, (uint64_t) x.suh_g, (uint64_t) x.suh_u, (uint64_t) sc.xg, (uint64_t) sc.xu, D, slots, E, r,
             (uint64_t) sc.ids, (uint64_t) sc.count, (uint64_t) sc.members, r, (uint64_t) dwts, (uint64_t) sc.y, 0, (uint64_t) dout, D, 1,
             (uint64_t) sc.epoch};
    const int blocks = 1 + (p * (D / 128) * 2 + 7) / 8;
    static hipFunction_t fprep = fn("tf_ds_exl3_experts_decode_prep");
    launch(fprep, dim3(blocks), dim3(256), (unsigned) (2 * p * 4), pp);
    if (upto < 2) return;
    CpP gu{(uint64_t) sc.xg, (uint64_t) sc.xu, (uint64_t) x.gate_ptr, (uint64_t) x.up_ptr, (uint64_t) x.gate_k2, (uint64_t) x.up_k2,
           (uint64_t) sc.ids, (uint64_t) sc.count, (uint64_t) sc.members, (uint64_t) sc.z, D, I, p, cgu[2], r, slots, {}};
    memset(&gu.epi, 0, sizeof gu.epi);
    gu.epi.pick = (uint64_t) dpick; gu.epi.E = E; gu.epi.svh_g = (uint64_t) x.svh_g; gu.epi.svh_u = (uint64_t) x.svh_u;
    gu.epi.suh_d = (uint64_t) x.suh_d; gu.epi.xd = (uint64_t) sc.xd; gu.epi.limit = limit; gu.epi.act_mode = 1; gu.epi.store_y = 1;
    gu.epi.cnt = (uint64_t) sc.cnt_gu; gu.epi.ready = (uint64_t) sc.ready; gu.epi.ready_cnt = (uint64_t) sc.ready_cnt;
    gu.epi.epoch = (uint64_t) sc.epoch; gu.epi.discard = 8 * cgu[2] <= 32;
    const std::string rg = range_name(x.k2_gu[0], x.k2_gu[1]), rd = range_name(x.k2_d[0], x.k2_d[1]);
    if (g_fused && upto >= 3)
    {
        CpP dn{(uint64_t) sc.xd, (uint64_t) sc.xd, (uint64_t) x.down_ptr, (uint64_t) x.down_ptr, (uint64_t) x.down_k2, (uint64_t) x.down_k2,
               (uint64_t) sc.ids, (uint64_t) sc.count, (uint64_t) sc.members, (uint64_t) sc.z, I, D, p, 1, r, slots, {}};
        memset(&dn.epi, 0, sizeof dn.epi);
        dn.epi.pick = (uint64_t) dpick; dn.epi.E = E; dn.epi.svh_d = (uint64_t) x.svh_d; dn.epi.y = (uint64_t) sc.y; dn.epi.wts = (uint64_t) dwts;
        dn.epi.out = (uint64_t) dout; dn.epi.act_mode = 1; dn.epi.store_y = 1; dn.epi.cnt = (uint64_t) sc.cnt_d; dn.epi.ready = (uint64_t) sc.ready;
        dn.epi.ready_cnt = (uint64_t) sc.ready_cnt; dn.epi.epoch = (uint64_t) sc.epoch; dn.epi.discard = 4 * slots <= 32;
        const dim3 ggu(maxu, I / 128, 2 * cgu[2] * mt), gdn(maxu, D / 128, mt);
        const CpFusedP f{cp_launch(gu, ggu), cp_launch(dn, gdn)};
        const std::string rr = range_name(std::min(x.k2_gu[0], x.k2_d[0]), std::max(x.k2_gu[1], x.k2_d[1]));
        launch(fn("tf_ds_exl3_experts_grouped_cpf_" + rr), dim3(ggu.x * ggu.y * ggu.z + gdn.x * gdn.y * gdn.z), dim3(128), 0, f);
        return;
    }
    const bool split = g_lean && g_split && r > 1;
    const std::string cp = g_lean ? "tf_ds_exl3_experts_grouped_cp1_" : "tf_ds_exl3_experts_grouped_cp_";
    if (r > 1 && !split) {
        launch(fn("tf_ds_exl3_experts_grouped_cp_" + rg + "_epi1"), dim3(maxu, I / 128, 2 * cgu[2] * mt), dim3(128), 0, gu);
    } else {
        launch(fn(cp + rg + "_epi1"), dim3(maxu, I / 128, 2 * cgu[2] * mt), dim3(128), 0, gu);
        if (split) launch(fn("tf_ds_exl3_experts_grouped_cpm_" + rg + "_epi1"), dim3(maxu, I / 128, 2 * cgu[2] * mt), dim3(128), 0, gu);
    }
    if (upto < 3) return;
    CpP dn{(uint64_t) sc.xd, (uint64_t) sc.xd, (uint64_t) x.down_ptr, (uint64_t) x.down_ptr, (uint64_t) x.down_k2, (uint64_t) x.down_k2,
           (uint64_t) sc.ids, (uint64_t) sc.count, (uint64_t) sc.members, (uint64_t) sc.z, I, D, p, 1, r, slots, {}};
    memset(&dn.epi, 0, sizeof dn.epi);
    dn.epi.pick = (uint64_t) dpick; dn.epi.E = E; dn.epi.svh_d = (uint64_t) x.svh_d; dn.epi.y = (uint64_t) sc.y; dn.epi.wts = (uint64_t) dwts;
    dn.epi.out = (uint64_t) dout; dn.epi.act_mode = 1; dn.epi.store_y = 1; dn.epi.cnt = (uint64_t) sc.cnt_d; dn.epi.ready = (uint64_t) sc.ready;
    dn.epi.ready_cnt = (uint64_t) sc.ready_cnt; dn.epi.epoch = (uint64_t) sc.epoch; dn.epi.discard = 4 * slots <= 32;
    if (r > 1 && !split) {
        launch(fn("tf_ds_exl3_experts_grouped_cp_" + rd + "_epi2"), dim3(maxu, D / 128, mt), dim3(128), 0, dn);
    } else {
        launch(fn(cp + rd + "_epi2"), dim3(maxu, D / 128, mt), dim3(128), 0, dn);
        if (split) launch(fn("tf_ds_exl3_experts_grouped_cpm_" + rd + "_epi2"), dim3(maxu, D / 128, mt), dim3(128), 0, dn);
    }
}

static int mode_dexperts(int R, int layer)
{
    const int routed = env_int("ROUTED", 384), slots = 7;
    const float limit = 10.0f;
    EExperts x = load_eexperts(layer, routed);
    const int D = x.D;
    DecodeScratch sc = alloc_decode(std::max(R, env_int("SCRATCH_ROWS", 64)), slots, D, x.I, x.E);
    std::vector<uint16_t> xh16 = rand_x((size_t) R * D), xb(xh16.size());
    for (size_t j = 0; j < xb.size(); ++j) xb[j] = f_to_bf((float) exo_f16_to_f64(xh16[j]));
    uint16_t* dx = dupload(xb);
    float* dout = dalloc<float>((size_t) R * D);
    const int ITER_ROUTES = 64;
    std::vector<int32_t*> dpick(ITER_ROUTES);
    std::vector<float*> dwts(ITER_ROUTES);
    std::vector<std::vector<int32_t>> picks(ITER_ROUTES);
    std::vector<std::vector<float>> wtss(ITER_ROUTES);
    for (int i = 0; i < ITER_ROUTES; ++i)
    {
        make_routes(R, slots, routed, i, picks[i], wtss[i]);
        dpick[i] = dupload(picks[i]);
        dwts[i] = dupload(wtss[i]);
    }
    // correctness: two consecutive calls (epoch 1, 2), both graded
    std::vector<int> rows(R);
    std::iota(rows.begin(), rows.end(), 0);
    XGrade worst;
    worst.pass = true;
    for (int call = 0; call < 2; ++call)
    {
        decode_experts(x, sc, dx, dpick[call], dwts[call], dout, R, limit);
        HC(hipStreamSynchronize(g_stream));
        auto got = download(dout, (size_t) R * D);
        auto ref = ref_experts(x, xb, picks[call], wtss[call], slots, rows, limit);
        XGrade g = xgrade(got, ref, R, D);
        worst.nrmse = std::max(worst.nrmse, g.nrmse); worst.worst = std::max(worst.worst, g.worst);
        worst.nonfinite += g.nonfinite; worst.pass = worst.pass && g.pass;
    }
    // the scratch protocol after two calls: epoch 2, the used places' ready = 2, every counter zero
    const int ep = download(sc.epoch, 1)[0];
    const int nu = download(sc.count, 1)[0];
    auto rdy = download(sc.ready, std::max(1, sc.maxu));
    size_t ready_bad = 0, cnt_nonzero = 0;
    for (int u = 0; u < nu; ++u) ready_bad += rdy[u] != ep;
    for (int v : download(sc.ready_cnt, std::max(1, sc.maxu))) cnt_nonzero += v != 0;
    for (int v : download(sc.cnt_gu, (size_t) sc.maxu * (x.I / 128))) cnt_nonzero += v != 0;
    for (int v : download(sc.cnt_d, (size_t) sc.rows * (D / 128))) cnt_nonzero += v != 0;
    // timing: fresh routes each iteration (cold experts), bytes = the distinct experts' trellises
    double bytes = 0;
    for (int i = 0; i < ITER_ROUTES; ++i)
    {
        std::vector<int> seen;
        for (int v : picks[i]) if (std::find(seen.begin(), seen.end(), v) == seen.end()) seen.push_back(v);
        for (int e : seen) bytes += x.bytes_of(e);
    }
    bytes /= ITER_ROUTES;
    const int iters = env_int("ITERS", 100), warm = 10;
    auto run = [&](int it) { decode_experts(x, sc, dx, dpick[it % ITER_ROUTES], dwts[it % ITER_ROUTES], dout, R, limit); };
    Stats st = time_events(run, warm, iters);
    const double b2b = time_b2b(run, warm, iters);
    const double gus = time_graph(run, ITER_ROUTES / 2, env_int("GRAPH_REPS", 20));
    // stage breakdown (back to back): decode_prep alone, + gate/up, all three
    double up_us[3];
    for (int u = 1; u <= 3; ++u)
        up_us[u - 1] = time_b2b([&](int it) { decode_experts(x, sc, dx, dpick[it % ITER_ROUTES], dwts[it % ITER_ROUTES], dout, R, limit, u); }, warm, iters);
    fprintf(stderr, "stages us: prep %.1f gate_up %.1f down %.1f\n", up_us[0], up_us[1] - up_us[0], up_us[2] - up_us[1]);
    const bool ok = worst.pass && ep == 2 && ready_bad == 0 && cnt_nonzero == 0;
    printf("{\"mode\":\"dexperts\",\"layer\":%d,\"rows\":%d,\"slots\":%d,\"experts\":%d,\"k2_gu\":[%d,%d],\"k2_d\":[%d,%d],\"places\":%d,"
           "\"bytes_mean\":%.0f,\"median_us\":%.2f,\"b2b_us\":%.2f,\"graph_us\":%.2f,\"gbs_median\":%.1f,\"gbs_b2b\":%.1f,\"sclk_min\":%d,\"sclk_med\":%d,\"load1\":%.2f,"
           "\"stage_us\":{\"prep\":%.1f,\"gate_up\":%.1f,\"down\":%.1f},\"epoch\":%d,\"ready_bad\":%zu,\"counters_nonzero\":%zu,\"nrmse\":%.3g,\"row_nrmse_max\":%.3g,\"nonfinite\":%zu,\"pass\":%s}\n",
           layer, R, slots, x.E, x.k2_gu[0], x.k2_gu[1], x.k2_d[0], x.k2_d[1], nu, bytes, st.median_us, b2b, gus, bytes / (st.median_us * 1e3),
           bytes / (b2b * 1e3), g_clk_min, g_clk_med, g_load1, up_us[0], up_us[1] - up_us[0], up_us[2] - up_us[1], ep, ready_bad, cnt_nonzero, worst.nrmse, worst.worst, worst.nonfinite,
           ok ? "true" : "false");
    fflush(stdout);
    return ok ? 0 : 1;
}

// ---- prompt experts ---------------------------------------------------------------------------------------------------------

struct CountP { uint64_t pick, counts; int32_t p; };
struct PlaceP { uint64_t pick, counts, ids, count, members; int32_t p, slots, e, maxm; };
struct ListP { uint64_t counts, ids, count; int32_t maxu; uint64_t work_gu; int32_t rows_gu, n_gu; uint64_t work_d; int32_t rows_d, n_d, skip_e; };
struct XRotP { uint64_t x; int32_t x_stride; uint64_t pick, suh_g, suh_u, xg, xu; int32_t d, slots, e; };
struct MmaP { uint64_t xa, xb, wa, wb, ka, kb, ids, count, members, z; int32_t k, n, p, splits, warps, maxm, slots, n_work; uint64_t work; };
struct EpiP { uint64_t z, pick, svh_g, svh_u, suh_d, xd; int32_t p, i, splits, e; float limit; int32_t act; };
struct RowsP { uint64_t xa, xb, wa, wb, ka, kb, ids, count, members, z; int32_t k, n, p, splits, maxm, slots, n_work; uint64_t work; };
struct CombP { uint64_t z, pick, svh_d, y, wts, add, out; int32_t p, d, splits, e, slots, store_y; };
static_assert(offsetof(MmaP, work) == 112 && offsetof(RowsP, work) == 112 && offsetof(ListP, skip_e) == 64, "prompt packing");

static int list_len(int p, int maxu, int rows) { return (p + maxu * (rows - 1)) / rows; }
static int member_stride(int r) { return std::max(32, (r + 31) / 32 * 32); }

struct PromptScratch
{
    int rows, slots, maxu, maxm, n_gu, n_d;
    uint16_t *xg, *xu, *xd;
    float *z, *no_y;
    int32_t *ids, *count, *counts, *members, *work_gu, *work_d;
};

static PromptScratch alloc_prompt(int rows, int slots, int D, int I, int E)
{
    PromptScratch s;
    s.rows = rows; s.slots = slots;
    const int p = rows * slots;
    s.maxu = std::min(p, E);
    s.maxm = member_stride(rows);
    s.n_gu = list_len(p, s.maxu, 64);
    s.n_d = list_len(p, s.maxu, 32);
    const size_t zf = (size_t) std::max(2 * I + D / 2, D) * p;
    s.xg = dalloc<uint16_t>((size_t) p * D); dzero(s.xg, (size_t) p * D * 2);
    s.xu = dalloc<uint16_t>((size_t) p * D); dzero(s.xu, (size_t) p * D * 2);
    s.xd = dalloc<uint16_t>((size_t) p * I); dzero(s.xd, (size_t) p * I * 2);
    s.z = dalloc<float>(zf); dzero(s.z, zf * 4);
    s.no_y = dalloc<float>(1); dzero(s.no_y, 4);
    s.ids = dalloc<int32_t>(s.maxu); dzero(s.ids, s.maxu * 4);
    s.count = dalloc<int32_t>(1); dzero(s.count, 4);
    s.counts = dalloc<int32_t>(E); dzero(s.counts, E * 4);
    const size_t mb = (size_t) s.maxu * ((rows + 127) / 128 * 128);
    s.members = dalloc<int32_t>(mb); dfill(s.members, 0xff, mb * 4);
    s.work_gu = dalloc<int32_t>((size_t) 2 * s.n_gu);
    s.work_d = dalloc<int32_t>((size_t) 2 * s.n_d);
    return s;
}

static void prompt_experts(const EExperts& x, const PromptScratch& sc, const uint16_t* dx, const int32_t* dpick, const float* dwts,
                           float* dout, int r, float limit, int stages = 31)
{
    const int D = x.D, I = x.I, E = x.E, slots = sc.slots, p = r * slots;
    const auto cgu = xconfig(D, I, true);
    static hipFunction_t fcount = fn("tf_ds_exl3_experts_group_count"), fplace = fn("tf_ds_exl3_experts_group_place"),
                         flist = fn("tf_ds_exl3_experts_work_list"), frot = fn("tf_ds_exl3_experts_rot_in"),
                         fepi = fn("tf_ds_exl3_experts_gateup_epilogue"), fcomb = fn("tf_ds_exl3_experts_down_combine");
    if (stages & 1)
    {
        launch(fcount, dim3(E), dim3(256), 0, CountP{(uint64_t) dpick, (uint64_t) sc.counts, p});
        launch(fplace, dim3(E), dim3(256), 0, PlaceP{(uint64_t) dpick, (uint64_t) sc.counts, (uint64_t) sc.ids, (uint64_t) sc.count, (uint64_t) sc.members, p, slots, E, sc.maxm});
        launch(flist, dim3(1), dim3(1024), 0, ListP{(uint64_t) sc.counts, (uint64_t) sc.ids, (uint64_t) sc.count, sc.maxu, (uint64_t) sc.work_gu, 64, sc.n_gu, (uint64_t) sc.work_d, 32, sc.n_d, -1});
        launch(frot, dim3(p, D / 128, 2), dim3(32), 0, XRotP{(uint64_t) dx, D, (uint64_t) dpick, (uint64_t) x.suh_g, (uint64_t) x.suh_u, (uint64_t) sc.xg, (uint64_t) sc.xu, D, slots, E});
    }
    if (stages & 2)
        launch(fn("tf_ds_exl3_experts_grouped_mma_" + range_name(x.k2_gu[0], x.k2_gu[1])), dim3(1, I / 64, sc.n_gu * 2), dim3(256), 0,
               MmaP{(uint64_t) sc.xg, (uint64_t) sc.xu, (uint64_t) x.gate_ptr, (uint64_t) x.up_ptr, (uint64_t) x.gate_k2, (uint64_t) x.up_k2, (uint64_t) sc.ids,
                    (uint64_t) sc.count, (uint64_t) sc.members, (uint64_t) sc.z, D, I, p, cgu[2], cgu[1], sc.maxm, slots, sc.n_gu, (uint64_t) sc.work_gu});
    if (stages & 4)
        launch(fepi, dim3(p, I / 128), dim3(32), 0, EpiP{(uint64_t) sc.z, (uint64_t) dpick, (uint64_t) x.svh_g, (uint64_t) x.svh_u, (uint64_t) x.suh_d, (uint64_t) sc.xd, p, I, 1, E, limit, 1});
    if (stages & 8)
        launch(fn("tf_ds_exl3_experts_grouped_rows_" + range_name(x.k2_d[0], x.k2_d[1])), dim3(1, D / 128, sc.n_d), dim3(128), 0,
               RowsP{(uint64_t) sc.xd, (uint64_t) sc.xd, (uint64_t) x.down_ptr, (uint64_t) x.down_ptr, (uint64_t) x.down_k2, (uint64_t) x.down_k2, (uint64_t) sc.ids,
                     (uint64_t) sc.count, (uint64_t) sc.members, (uint64_t) sc.z, I, D, p, 1, sc.maxm, slots, sc.n_d, (uint64_t) sc.work_d});
    if (stages & 16)
        launch(fcomb, dim3(r, D / 128), dim3(32 * slots), 0, CombP{(uint64_t) sc.z, (uint64_t) dpick, (uint64_t) x.svh_d, (uint64_t) sc.no_y, (uint64_t) dwts, 0, (uint64_t) dout, p, D, 1, E, slots, 0});
}

static int mode_pexperts(int M, int layer)
{
    const int routed = env_int("ROUTED", 384), slots = 7;
    const float limit = 10.0f;
    EExperts x = load_eexperts(layer, routed);
    const int D = x.D;
    PromptScratch sc = alloc_prompt(M, slots, D, x.I, x.E);
    std::vector<uint16_t> xh16 = rand_x((size_t) M * D), xb(xh16.size());
    for (size_t j = 0; j < xb.size(); ++j) xb[j] = f_to_bf((float) exo_f16_to_f64(xh16[j]));
    uint16_t* dx = dupload(xb);
    float* dout = dalloc<float>((size_t) M * D);
    std::vector<int32_t> pick;
    std::vector<float> wts;
    make_routes(M, slots, routed, 0, pick, wts);
    int32_t* dpick = dupload(pick);
    float* dwts = dupload(wts);
    prompt_experts(x, sc, dx, dpick, dwts, dout, M, limit);
    HC(hipStreamSynchronize(g_stream));
    auto got = download(dout, (size_t) M * D);
    // grade a sample of rows (first, last, and spread)
    const int ns = std::min(M, env_int("SAMPLE_ROWS", 12));
    std::vector<int> rows;
    for (int i = 0; i < ns; ++i) rows.push_back((int) ((int64_t) i * (M - 1) / std::max(1, ns - 1)));
    auto ref = ref_experts(x, xb, pick, wts, slots, rows, limit);
    std::vector<float> gs((size_t) ns * D);
    for (int i = 0; i < ns; ++i) memcpy(&gs[(size_t) i * D], &got[(size_t) rows[i] * D], (size_t) D * 4);
    XGrade g = xgrade(gs, ref, ns, D);
    size_t nonfinite = 0;
    for (float v : got) nonfinite += !std::isfinite(v);
    // timing: the whole chunk and each stage
    const int iters = env_int("ITERS", 10), warm = 2;
    const double us = time_b2b([&](int) { prompt_experts(x, sc, dx, dpick, dwts, dout, M, limit); }, warm, iters);
    double st_us[5];
    for (int s = 0; s < 5; ++s) st_us[s] = time_b2b([&](int) { prompt_experts(x, sc, dx, dpick, dwts, dout, M, limit, 1 << s); }, warm, iters);
    // useful FLOPs: per (row, slot) 2 * 3 * D * I
    const double flops = 2.0 * 3.0 * D * x.I * (double) M * slots;
    const bool ok = g.pass && nonfinite == 0;
    printf("{\"mode\":\"pexperts\",\"layer\":%d,\"rows\":%d,\"slots\":%d,\"experts\":%d,\"k2_gu\":[%d,%d],\"k2_d\":[%d,%d],\"total_us\":%.0f,"
           "\"stage_us\":{\"route_rot\":%.0f,\"gateup\":%.0f,\"epilogue\":%.0f,\"down\":%.0f,\"combine\":%.0f},\"tflops\":%.2f,\"load1\":%.2f,"
           "\"sampled_rows\":%d,\"nrmse\":%.3g,\"row_nrmse_max\":%.3g,\"nonfinite\":%zu,\"pass\":%s}\n",
           layer, M, slots, x.E, x.k2_gu[0], x.k2_gu[1], x.k2_d[0], x.k2_d[1], us, st_us[0], st_us[1], st_us[2], st_us[3], st_us[4],
           flops / (us * 1e6), load1(), ns, g.nrmse, g.worst, nonfinite, ok ? "true" : "false");
    fflush(stdout);
    return ok ? 0 : 1;
}


// ---- the AMD kit's prompt experts (exl3_pf.hip): exl3_experts.zig pfPlan and promptPf ------------------------------------

struct PfPlan { tf_ds_exl3_pf_args a; int nm; std::vector<int> k2_gu, k2_d; unsigned grids[4]; };

// pfPlan, line for line: one gu and one down launch for each K2 of the table (weights.zig k2_masks), ascending
static PfPlan pf_plan(const EExperts& x, int r, int slots, const uint16_t* dx, const int32_t* dpick, const float* dwts, float* dout,
                      const PromptScratch& sc, int32_t* order, int32_t* jobs, int32_t* meta, float limit)
{
    const int D = x.D, I = x.I, E = x.E, p = r * slots;
    PfPlan q;
    unsigned mgu = 0, md = 0;
    for (int e = 0; e < E; ++e)
    {
        if (x.g[e].K != x.u[e].K) DIE("expert %d: gate and up bits differ (pfPlan would refuse)", e);
        mgu |= 1u << (2 * x.g[e].K);
        md |= 1u << (2 * x.d[e].K);
    }
    for (int k2 = 0; k2 <= 16; ++k2)
    {
        if (mgu >> k2 & 1) q.k2_gu.push_back(k2);
        if (md >> k2 & 1) q.k2_d.push_back(k2);
    }
    const int nm = p >= 48 * E ? 128 : 64;
    const int maxjobs = p / nm + E;
    tf_ds_exl3_pf_args& a = q.a;
    memset(&a, 0, sizeof a);
    a.gate_ptr = (uint64_t) x.gate_ptr; a.up_ptr = (uint64_t) x.up_ptr; a.down_ptr = (uint64_t) x.down_ptr;
    a.gate_k2 = (uint64_t) x.gate_k2; a.up_k2 = (uint64_t) x.up_k2; a.down_k2 = (uint64_t) x.down_k2;
    a.suh_g = (uint64_t) x.suh_g; a.suh_u = (uint64_t) x.suh_u; a.svh_g = (uint64_t) x.svh_g; a.svh_u = (uint64_t) x.svh_u;
    a.suh_d = (uint64_t) x.suh_d; a.svh_d = (uint64_t) x.svh_d;
    a.x = (uint64_t) dx; a.ids = (uint64_t) dpick; a.wts = (uint64_t) dwts;
    a.xg = (uint64_t) sc.xg; a.xu = (uint64_t) sc.xu; a.xd = (uint64_t) sc.xd; a.z = (uint64_t) sc.z;
    a.order = (uint64_t) order; a.jobs = (uint64_t) jobs; a.meta = (uint64_t) meta; a.out = (uint64_t) dout;
    a.r = r; a.slots = slots; a.e = E; a.d = D; a.i = I; a.ldx = D; a.ldo = D; a.maxjobs = maxjobs; a.nmax = nm; a.accumulate = 0;
    a.limit = limit; a.x_fmt = 1;
    q.nm = nm;
    const unsigned g[4] = {1u, (unsigned) ((E + 7) / 8), (unsigned) p, (unsigned) std::min((r * (D / 4) + 255) / 256, 8192)};
    memcpy(q.grids, g, sizeof g);
    return q;
}

// promptPf (stage mask: 1 count, 2 place, 4 rot, 8 gu, 16 down, 32 combine)
static void pf_run(const PfPlan& q, int stages = 63)
{
    const tf_ds_exl3_pf_args& a = q.a;
    const std::string nm = std::to_string(q.nm);
    if (stages & 1) launch(fn("tf_ds_exl3_pf_experts_count"), dim3(q.grids[0]), dim3(1024), 0, a);
    if (stages & 2) launch(fn("tf_ds_exl3_pf_experts_place"), dim3(q.grids[1]), dim3(256), 0, a);
    if (stages & 4) launch(fn("tf_ds_exl3_pf_experts_rot"), dim3(q.grids[2]), dim3(256), 0, a);
    if (stages & 8)
        for (int k2 : q.k2_gu) launch(fn("tf_ds_exl3_pf_experts_gu" + nm + "_k" + std::to_string(k2)), dim3(a.maxjobs * (a.i / 128)), dim3(256), 0, a);
    if (stages & 16)
        for (int k2 : q.k2_d) launch(fn("tf_ds_exl3_pf_experts_down" + nm + "_k" + std::to_string(k2)), dim3(a.maxjobs * (a.d / 256)), dim3(256), 0, a);
    if (stages & 32) launch(fn("tf_ds_exl3_pf_experts_combine"), dim3(q.grids[3]), dim3(256), 0, a);
}

// real router picks for the routed slots (ROUTE_FILE, '%d' = layer: int32 [M * (slots - 1)], w-ref's trace via w-pf), the
// shared expert last, make_routes' weights
static void load_routes(const char* rf, int layer, int M, int slots, std::vector<int32_t>& pick)
{
    std::string path = rf[0] == '/' ? std::string(rf) : ROOT + rf;
    if (strstr(rf, "%d")) { char b[512]; snprintf(b, sizeof b, path.c_str(), layer); path = b; }
    std::vector<int32_t> routed((size_t) M * (slots - 1));
    FILE* f = fopen(path.c_str(), "rb");
    if (!f || fread(routed.data(), 4, routed.size(), f) != routed.size()) DIE("ROUTE_FILE %s: need %zu int32", path.c_str(), routed.size());
    fclose(f);
    for (int r = 0; r < M; ++r)
        for (int s = 0; s < slots - 1; ++s) pick[(size_t) r * slots + s] = routed[(size_t) r * (slots - 1) + s];
}

static int mode_pfexperts(int M, int layer)
{
    const int routed = env_int("ROUTED", 384), slots = env_int("SLOTS", 7);
    const float limit = 10.0f;
    EExperts x = load_eexperts(layer, routed);
    const int D = x.D, E = x.E, p = M * slots;
    PromptScratch sc = alloc_prompt(M, slots, D, x.I, E);
    int32_t* order = dalloc<int32_t>((size_t) p);
    int32_t* jobs = dalloc<int32_t>((size_t) 3 * (p / 64 + E));
    int32_t* meta = dalloc<int32_t>(2 * 1025);
    std::vector<uint16_t> xh16 = rand_x((size_t) M * D), xb(xh16.size());
    for (size_t j = 0; j < xb.size(); ++j) xb[j] = f_to_bf((float) exo_f16_to_f64(xh16[j]));
    uint16_t* dx = dupload(xb);
    float* dout = dalloc<float>((size_t) M * D);
    float* dout2 = dalloc<float>((size_t) M * D);
    std::vector<int32_t> pick;
    std::vector<float> wts;
    make_routes(M, slots, routed, 0, pick, wts);
    const char* rf = getenv("ROUTE_FILE");
    if (rf) load_routes(rf, layer, M, slots, pick);
    int32_t* dpick = dupload(pick);
    float* dwts = dupload(wts);
    // out starts as garbage: the routed pass must overwrite every row
    dfill(dout, 0x7f, (size_t) M * D * 4);
    const PfPlan ps = pf_plan(x, M, slots, dx, dpick, dwts, dout, sc, order, jobs, meta, limit);
    pf_run(ps);
    prompt_experts(x, sc, dx, dpick, dwts, dout2, M, limit);   // the grouped path on the same routes (its own scratch fields)
    HC(hipStreamSynchronize(g_stream));
    auto got = download(dout, (size_t) M * D);
    auto got2 = download(dout2, (size_t) M * D);
    // graded rows: spread, plus a row of the most- and of the least-used routed expert
    std::vector<int> cnt(E, 0);
    for (int r = 0; r < M; ++r) for (int s = 0; s < slots - 1; ++s) cnt[pick[(size_t) r * slots + s]]++;
    int emax = 0, emin = -1;
    for (int e = 0; e < E - 1; ++e) { if (cnt[e] > cnt[emax]) emax = e; if (cnt[e] > 0 && (emin < 0 || cnt[e] < cnt[emin])) emin = e; }
    const int ns = std::min(M, env_int("SAMPLE_ROWS", 10));
    std::vector<int> rows;
    for (int i = 0; i < ns; ++i) rows.push_back((int) ((int64_t) i * (M - 1) / std::max(1, ns - 1)));
    for (int want : {emax, emin})
        for (int r = 0; r < M; ++r)
        {
            bool hit = false;
            for (int s = 0; s < slots - 1; ++s) hit |= pick[(size_t) r * slots + s] == want;
            if (hit) { if (std::find(rows.begin(), rows.end(), r) == rows.end()) rows.push_back(r); break; }
        }
    auto ref = ref_experts(x, xb, pick, wts, slots, rows, limit);
    std::vector<float> gs(rows.size() * D), gs2(rows.size() * D);
    for (size_t i = 0; i < rows.size(); ++i)
    {
        memcpy(&gs[i * D], &got[(size_t) rows[i] * D], (size_t) D * 4);
        memcpy(&gs2[i * D], &got2[(size_t) rows[i] * D], (size_t) D * 4);
    }
    XGrade g = xgrade(gs, ref, (int) rows.size(), D), g2 = xgrade(gs2, ref, (int) rows.size(), D);
    size_t nonfinite = 0;
    for (float v : got) nonfinite += !std::isfinite(v);
    // the two paths on every row (not a grade: both carry their own error against FP64)
    double se = 0, sr = 0;
    for (size_t j = 0; j < got.size(); ++j) { se += (double) (got[j] - got2[j]) * (got[j] - got2[j]); sr += (double) got2[j] * got2[j]; }
    // repeatability: a second call gives the same bits (no atomics in the sums)
    pf_run(ps);
    HC(hipStreamSynchronize(g_stream));
    auto again = download(dout, (size_t) M * D);
    const bool repeat = memcmp(again.data(), got.data(), got.size() * 4) == 0;
    const int iters = env_int("ITERS", 10), warm = 2;
    const double us = time_b2b([&](int) { pf_run(ps); }, warm, iters);
    double st_us[6];
    for (int s = 0; s < 6; ++s) st_us[s] = time_b2b([&](int) { pf_run(ps, 1 << s); }, warm, iters);
    const double us_grouped = time_b2b([&](int) { prompt_experts(x, sc, dx, dpick, dwts, dout2, M, limit); }, warm, iters);
    const double flops = 2.0 * 3.0 * D * x.I * (double) M * slots;
    const bool ok = g.pass && nonfinite == 0 && repeat;
    std::string k2s = "[";
    for (int k2 : ps.k2_gu) k2s += (k2s.size() > 1 ? "," : "") + std::to_string(k2);
    k2s += "],[";
    for (size_t j = 0; j < ps.k2_d.size(); ++j) k2s += (j ? "," : "") + std::to_string(ps.k2_d[j]);
    k2s += "]";
    printf("{\"mode\":\"pfexperts\",\"layer\":%d,\"mtp\":%s,\"rows\":%d,\"slots\":%d,\"experts\":%d,\"routes\":\"%s\",\"k2_gu_d\":[%s],\"nm\":%d,"
           "\"total_us\":%.0f,\"stage_us\":{\"count\":%.0f,\"place\":%.0f,\"rot\":%.0f,\"gu\":%.0f,\"down\":%.0f,\"combine\":%.0f},"
           "\"grouped_us\":%.0f,\"speedup\":%.2f,\"tflops\":%.2f,\"sclk_med\":%d,\"load1\":%.2f,\"graded_rows\":%zu,\"nrmse\":%.3g,\"row_nrmse_max\":%.3g,"
           "\"grouped_nrmse\":%.3g,\"grouped_row_nrmse_max\":%.3g,\"pf_vs_grouped_nrmse\":%.3g,\"nonfinite\":%zu,\"repeatable\":%s,\"pass\":%s}\n",
           layer, getenv("MTP") ? "true" : "false", M, slots, E, rf ? rf : "uniform", k2s.c_str(), ps.nm,
           us, st_us[0], st_us[1], st_us[2], st_us[3], st_us[4], st_us[5], us_grouped, us_grouped / us, flops / (us * 1e6),
           read_sclk(), load1(), rows.size(), g.nrmse, g.worst, g2.nrmse, g2.worst, sqrt(se / std::max(sr, 1e-300)), nonfinite, repeat ? "true" : "false",
           ok ? "true" : "false");
    fflush(stdout);
    return ok ? 0 : 1;
}

// ---- the dense prompt path: rot_in, unpack, then w-tri's Triton _gemm (exl3_prefill.zig matmul) -------------------------

// tri_markov.zig matmul's _gemm launch: X, W, H, SVH, BIAS, OUT, M, o_stride (+ Triton's two hidden scratch pointers)
struct GemmP { uint64_t x, w, h, svh, bias, out; int32_t m, o_stride; uint64_t gscratch, pscratch; };
static_assert(sizeof(GemmP) == 72 && offsetof(GemmP, gscratch) == 56, "_gemm packing");
// exl3_prefill.zig gemm: exl3.gemm's parameter list (hip_load.params.Exl3Gemm)
struct XGemmP { uint64_t xh, w; int64_t stride_k, stride_nb; uint64_t svh, out; int64_t o_stride; int32_t m, k, n, out_f32; };
static_assert(sizeof(XGemmP) == 72 && offsetof(XGemmP, m) == 56, "exl3.gemm packing");

static int mode_pgemm(int M)
{
    struct V { int k, n; std::string file; int shared, f32; };
    std::vector<V> vars;
    {
        FILE* f = fopen((ROOT + "tmp/gemm_variants.tsv").c_str(), "r");
        if (!f) DIE("no tmp/gemm_variants.tsv");
        char file[512];
        int k, n, sh, blk, f32;
        while (fscanf(f, "%d %d %511s %d %d %d", &k, &n, file, &sh, &blk, &f32) == 6) vars.push_back({k, n, file, sh, f32});
        fclose(f);
    }
    const std::string kit = getenv("TRI_KIT") ? getenv("TRI_KIT") : std::string(getenv("HOME")) + "/tfp/tri/kit/gfx1151/";
    struct Case { std::string tensor; int slice, a, b; };
    const std::vector<Case> cases = {
        {"layers.20.attn.wq_a", 0, 0, 0}, {"layers.20.attn.wkv", 0, 0, 0}, {"layers.20.attn.wq_b", 1, 0, 16384},
        {"layers.20.attn.indexer.wq_b", 0, 0, 0}, {"layers.39.attn.wo_a.slice.0", 0, 0, 0}, {"layers.39.attn.wo_b", 2, 0, 4096},
        {"layers.20.attn.indexer.wk", 0, 0, 0}, {"layers.1.engram.wkv", 2, 0, 3072},
        // k-dec2 (receipt): the K2 12 layer 0 linears, Engram layer 14's K2 8, the compressor
        {"layers.0.attn.wq_a", 0, 0, 0}, {"layers.0.attn.wkv", 0, 0, 0}, {"layers.14.engram.wkv", 2, 0, 3072},
        {"layers.20.attn.compressor.wkv", 0, 0, 0},
    };
    std::vector<uint16_t> had(128 * 128);
    for (int i = 0; i < 128; ++i) for (int j = 0; j < 128; ++j) had[i * 128 + j] = (__builtin_popcount(i & j) & 1) ? 0xbf80 : 0x3f80;
    uint16_t* dh = dupload(had);
    int rc = 0;
    const char* only = getenv("ONLY");
    for (const Case& c : cases)
    {
        if (only && c.tensor.find(only) == std::string::npos) continue;
        Lin l = make_lin(c.tensor, c.slice, c.a, c.b);
        const V* v = nullptr;
        for (auto& x : vars) if (x.k == l.in && x.n == l.out) v = &x;
        // no Triton _gemm variant for the shape (k-dec2): the kit's exl3.gemm alone is graded (out as the engine's)
        const bool tri = v != nullptr;
        static V none{0, 0, "", 0, 0};
        if (!tri) { none = V{l.in, l.out, "", 0, env_int("F32_OUT", 0)}; v = &none; }
        hipModule_t gm = nullptr;
        hipFunction_t fg = nullptr;
        if (tri) { HC(hipModuleLoad(&gm, (kit + v->file).c_str())); HC(hipModuleGetFunction(&fg, gm, "_gemm")); }
        ELin e = eupload(l);
        std::vector<uint16_t> xh16 = rand_x((size_t) M * l.in), xb(xh16.size());
        for (size_t j = 0; j < xb.size(); ++j) xb[j] = f_to_bf((float) exo_f16_to_f64(xh16[j]));
        uint16_t* dx = dupload(xb);
        uint16_t* dxh = dalloc<uint16_t>(xb.size());
        uint16_t* dw = dalloc<uint16_t>((size_t) l.in * l.out);
        void* dout = v->f32 ? (void*) dalloc<float>((size_t) M * l.out) : (void*) dalloc<uint16_t>((size_t) M * l.out);
        const RotInP rp{(uint64_t) dx, 1, l.in, (uint64_t) e.suh, (uint64_t) dxh, l.in};
        const UnpackP up{(uint64_t) e.words, (uint64_t) dw, l.out, e.stride_k(), e.stride_nb()};
        hipFunction_t fr = fn("tf_ds_exl3_rot_in"), fu = fn("tf_ds_exl3_unpack_k" + std::to_string(e.k2));
        const GemmP gp{(uint64_t) dxh, (uint64_t) dw, (uint64_t) dh, (uint64_t) e.svh, (uint64_t) e.svh, (uint64_t) dout, M, l.out, 0, 0};
        auto run = [&](int stages) {
            if (stages & 1) launch(fr, dim3((l.in / 128 + 3) / 4, M), dim3(128), 0, rp);
            if (stages & 2) launch(fu, dim3(l.out / 16, l.in / 16), dim3(32), 0, up);
            if (stages & 4) launch(fg, dim3(((M + 127) / 128) * (l.out / 128)), dim3(256), (unsigned) v->shared, gp);
        };
        if (tri) run(7); else run(1);
        HC(hipStreamSynchronize(g_stream));
        std::vector<float> got((size_t) M * l.out);
        if (v->f32) got = download((const float*) dout, got.size());
        else { auto gb = download((const uint16_t*) dout, got.size()); for (size_t j = 0; j < gb.size(); ++j) got[j] = bf_to_f(gb[j]); }
        // grade sampled rows against w-ref's contract with bf16 output rounding (u_out 2^-8)
        const int ns = std::min(M, env_int("SAMPLE_ROWS", 16));
        std::vector<double> xd((size_t) ns * l.in);
        std::vector<float> gs((size_t) ns * l.out);
        for (int i = 0; i < ns; ++i)
        {
            const int r = (int) ((int64_t) i * (M - 1) / std::max(1, ns - 1));
            for (int k = 0; k < l.in; ++k) xd[(size_t) i * l.in + k] = bf_to_f(xb[(size_t) r * l.in + k]);
            for (int j = 0; j < l.out; ++j) gs[(size_t) i * l.out + j] = got[(size_t) r * l.out + j];
        }
        std::vector<uint16_t> wq = cpu_decode(l);
        std::vector<double> wy, wB, wP;
        wref_ref(l, ns, xd, wq, wy, wB, wP, v->f32 ? 0x1p-24 : 0x1p-8);
        // numerics.json 1.6.0 exl3.linear statistical gate: nrmse <= 1e-3 (fp16 xh), a bf16 output adds 2.5e-3
        const double out_gate = v->f32 ? 1e-3 : 3.5e-3;
        Contract ct = tri ? grade(gs, wy, wB, wP, ns, l.out, out_gate) : Contract{};
        // the GEMM's own error with a bf16 output: against the reference rounded to bf16 as stored
        double se = 0, sr = 0;
        for (size_t j = 0; j < wy.size(); ++j)
        {
            const double r = v->f32 ? wy[j] : (double) bf_to_f(f_to_bf((float) wy[j]));
            se += (gs[j] - r) * (gs[j] - r); sr += wy[j] * wy[j];
        }
        const double nrmse_stored = sqrt(se / std::max(sr, 1e-300));
        // exl3.gemm on the same rows (rot_in's xh), the tile exl3_prefill.gemmTile picks
        const int tile = (l.out / 128) * ((M + 255) / 256) >= 120 || (M >= 512 && 2 * ((M + 255) / 256) == (M + 127) / 128) ? 256 : 128;   // exl3_prefill.gemmTile (37247a2)
        hipFunction_t fx = fn("tf_ds_exl3_gemm_k" + std::to_string(e.k2) + "_m" + std::to_string(tile));
        void* dout2 = v->f32 ? (void*) dalloc<float>((size_t) M * l.out) : (void*) dalloc<uint16_t>((size_t) M * l.out);
        const XGemmP xp{(uint64_t) dxh, (uint64_t) e.words, e.stride_k(), e.stride_nb(), (uint64_t) e.svh, (uint64_t) dout2, l.out, M, l.in, l.out, v->f32};
        auto run_x = [&](int stages) {
            if (stages & 1) launch(fr, dim3((l.in / 128 + 3) / 4, M), dim3(128), 0, rp);
            if (stages & 4) launch(fx, dim3((l.out / 128) * ((M + tile - 1) / tile)), dim3(256), 0, xp);
        };
        run_x(5);
        HC(hipStreamSynchronize(g_stream));
        std::vector<float> got2((size_t) M * l.out);
        if (v->f32) got2 = download((const float*) dout2, got2.size());
        else { auto gb = download((const uint16_t*) dout2, got2.size()); for (size_t j = 0; j < gb.size(); ++j) got2[j] = bf_to_f(gb[j]); }
        std::vector<float> gs2((size_t) ns * l.out);
        for (int i = 0; i < ns; ++i)
        {
            const int r = (int) ((int64_t) i * (M - 1) / std::max(1, ns - 1));
            for (int j = 0; j < l.out; ++j) gs2[(size_t) i * l.out + j] = got2[(size_t) r * l.out + j];
        }
        Contract ct2 = grade(gs2, wy, wB, wP, ns, l.out, out_gate);
        double se2 = 0, sr2 = 0;
        for (size_t j = 0; j < wy.size(); ++j)
        {
            const double r = v->f32 ? wy[j] : (double) bf_to_f(f_to_bf((float) wy[j]));
            se2 += (gs2[j] - r) * (gs2[j] - r); sr2 += wy[j] * wy[j];
        }
        const int iters = env_int("ITERS", 20), warm = 3;
        double us[3];
        for (int s = 0; s < 3; ++s) us[s] = tri || s == 0 ? time_b2b([&](int) { run(1 << s); }, warm, iters) : 0.0;
        const double all = tri ? time_b2b([&](int) { run(7); }, warm, iters) : 0.0;
        const double us_x = time_b2b([&](int) { run_x(4); }, warm, iters);
        const double all_x = time_b2b([&](int) { run_x(5); }, warm, iters);
        const double fl = 2.0 * M * l.in * (double) l.out;
        printf("{\"mode\":\"pgemm\",\"triton_variant\":%s,\"tensor\":\"%s\",\"K\":%d,\"k\":%d,\"n\":%d,\"rows\":%d,\"rot_in_us\":%.1f,\"unpack_us\":%.1f,"
               "\"gemm_us\":%.1f,\"total_us\":%.1f,\"gemm_tflops\":%.2f,\"path_tflops\":%.2f,\"sclk_med\":%d,\"load1\":%.2f,\"sampled_rows\":%d,"
               "\"out\":\"%s\",\"nrmse_vs_stored_ref\":%.3g,",
               tri ? "true" : "false", c.tensor.c_str(), l.K, l.in, l.out, M, us[0], us[1], us[2], all, fl / (us[2] * 1e6), fl / (all * 1e6), read_sclk(), load1(), ns,
               v->f32 ? "fp32" : "bf16", nrmse_stored);
        print_contract(ct);
        printf(",\"xgemm\":{\"tile\":%d,\"gemm_us\":%.1f,\"total_us\":%.1f,\"gemm_tflops\":%.2f,\"path_tflops\":%.2f,\"nrmse_vs_stored_ref\":%.3g,",
               tile, us_x, all_x, fl / (us_x * 1e6), fl / (all_x * 1e6), sqrt(se2 / std::max(sr2, 1e-300)));
        print_contract(ct2);
        printf("}}\n");
        fflush(stdout);
        rc |= (tri ? ct.pass : true) && ct2.pass ? 0 : 1;
        if (gm) HC(hipModuleUnload(gm));
        for (void* p : g_allocs) HC(hipFree(p));
        g_allocs.clear();
        dh = dupload(had);
    }
    return rc;
}

// ==== k-dec2: exl3.l2_prefetch (the MALL prefetch kernel) =========================================================
// l2pf MB: the kit's tf_ds_exl3_l2_prefetch over MB MB in three ranges (unaligned tails), grid 40 x 256 as the launcher:
// its time unpaced and paced (wave_ns 4,559: 230 GB/s), that it writes nothing (the buffer's checksum and a guard
// region), and an NT read of the ranges right after it against the same read cold (256 MB flushed between).
__global__ void __launch_bounds__(256) l2pf_ntread(const mall_v4u* __restrict__ p, size_t n16, uint32_t* out)
{
    uint32_t acc = 0;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) { const mall_v4u v = __builtin_nontemporal_load(p + i); acc += v.x ^ v.w; }
    if (acc == 0x12345678u) out[0] = acc;
}
struct L2pfP { tf_ds_prefetch a; int64_t delay_ns; int64_t wave_ns; };
static_assert(sizeof(L2pfP) == 280 && offsetof(L2pfP, delay_ns) == 264, "L2pfP");

static int mode_l2pf(int mb, const std::string&)
{
    const size_t bytes = (size_t) mb << 20, guard = 1 << 20;
    char* buf = dalloc<char>(bytes + 2 * guard);
    std::vector<uint8_t> h(bytes + 2 * guard);
    for (size_t i = 0; i < h.size(); ++i) h[i] = (uint8_t) (i * 2654435761u >> 13);
    HC(hipMemcpy(buf, h.data(), h.size(), hipMemcpyHostToDevice));
    char* flushbuf = dalloc<char>((size_t) 256 << 20);
    uint32_t* sink = dalloc<uint32_t>(4);
    // three ranges: a third each, the first two cut at odd byte counts (the kernel reads whole 16-byte units)
    const size_t r0 = bytes / 3 + 7, r1 = bytes / 3 + 1000, r2 = bytes - r0 - r1;
    L2pfP p;
    memset(&p, 0, sizeof p);
    p.a.n = 3;
    p.a.ptr[0] = (uint64_t) (buf + guard); p.a.bytes[0] = r0;
    p.a.ptr[1] = (uint64_t) (buf + guard + r0); p.a.bytes[1] = r1;
    p.a.ptr[2] = (uint64_t) (buf + guard + r0 + r1); p.a.bytes[2] = r2;
    hipFunction_t f = fn("tf_ds_exl3_l2_prefetch");
    auto flush = [&]() { mall_touch<<<640, 256, 0, g_stream>>>((const mall_v4u*) flushbuf, ((size_t) 256 << 20) / 16, sink); };
    hipEvent_t a, b, c;
    HC(hipEventCreate(&a)); HC(hipEventCreate(&b)); HC(hipEventCreate(&c));
    double t_pf[2] = {0, 0}, t_hot = 0, t_cold = 0;
    for (int paced = 0; paced < 2; ++paced)
    {
        std::vector<float> pf, hot, cold;
        p.delay_ns = 0;
        p.wave_ns = paced ? 4559 : 0;
        for (int it = 0; it < 14; ++it)
        {
            flush();
            HC(hipEventRecord(a, g_stream));
            launch(f, dim3(40), dim3(256), 0, p);
            HC(hipEventRecord(b, g_stream));
            l2pf_ntread<<<640, 256, 0, g_stream>>>((const mall_v4u*) (buf + guard), bytes / 16, sink);
            HC(hipEventRecord(c, g_stream));
            HC(hipEventSynchronize(c));
            float x, y; HC(hipEventElapsedTime(&x, a, b)); HC(hipEventElapsedTime(&y, b, c));
            flush();
            HC(hipEventRecord(a, g_stream));
            l2pf_ntread<<<640, 256, 0, g_stream>>>((const mall_v4u*) (buf + guard), bytes / 16, sink);
            HC(hipEventRecord(b, g_stream));
            HC(hipEventSynchronize(b));
            float z; HC(hipEventElapsedTime(&z, a, b));
            if (it >= 4) { pf.push_back(x * 1e3f); hot.push_back(y * 1e3f); cold.push_back(z * 1e3f); }
        }
        std::sort(pf.begin(), pf.end()); std::sort(hot.begin(), hot.end()); std::sort(cold.begin(), cold.end());
        t_pf[paced] = pf[pf.size() / 2];
        if (!paced) { t_hot = hot[hot.size() / 2]; t_cold = cold[cold.size() / 2]; }
    }
    std::vector<uint8_t> back(h.size());
    HC(hipMemcpy(back.data(), buf, back.size(), hipMemcpyDeviceToHost));
    size_t changed = 0;
    for (size_t i = 0; i < h.size(); ++i) changed += back[i] != h[i];
    const bool ok = changed == 0 && t_hot < 0.6 * t_cold;
    printf("{\"mode\":\"l2pf\",\"mb\":%d,\"ranges\":3,\"grid\":40,\"prefetch_us_unpaced\":%.1f,\"prefetch_us_paced\":%.1f,"
           "\"gbs_unpaced\":%.1f,\"gbs_paced\":%.1f,\"ntread_after_us\":%.1f,\"ntread_cold_us\":%.1f,\"bytes_changed\":%zu,\"pass\":%s}\n",
           mb, t_pf[0], t_pf[1], bytes / (t_pf[0] * 1e3), bytes / (t_pf[1] * 1e3), t_hot, t_cold, changed, ok ? "true" : "false");
    fflush(stdout);
    return ok ? 0 : 1;
}

// ==== k-dec: M1.exl3-exact receipt =================================================================================
// The kit's tile decode (exl3.unpack: decode_own / decode_pair, the code glinear and the experts decode with) against
// the CPU oracle (exo_decode, written from the format), bit for bit over every value of real tensors of each
// checkpoint K2 and of synthetic tensors that together reach all 65,536 MUL1 states: the stream's wraparound (value
// 0's window starts in the tile's last bits), tile placement (exo_placement), the words' byte order (little-endian
// int16 pairs) and sign bits (the trellis is int16). States counted with exo_tile_states.
static Lin synth_lin(const std::string& name, int k, int n, int K, uint64_t seed);

static int mode_exact(int K, const std::string& tensor)
{
    Lin l;
    const bool synth = tensor == "synth";
    if (synth)
    {
        // random tiles until every state appears (1,024 x 1,024: 1M values, ~65.5k states expected; grown if not)
        int n = 1024;
        for (int tries = 0;; ++tries)
        {
            l = synth_lin("synth.exact.k" + std::to_string(K), 1024, n, K, 9000 + 97 * K + tries);
            std::vector<uint8_t> seen(65536, 0);
            uint16_t st[256];
            const size_t tw = 16 * K, tiles = l.tr.size() / tw;
            for (size_t t = 0; t < tiles; ++t) { exo_tile_states(K, &l.tr[t * tw], st); for (int p = 0; p < 256; ++p) seen[st[p]] = 1; }
            size_t c = 0;
            for (uint8_t v : seen) c += v;
            if (c == 65536 || tries >= 4) break;
            n *= 2;
        }
    }
    else l = load_lin(tensor);
    if (l.K != K) DIE("%s is K %d, not %d", tensor.c_str(), l.K, K);
    // states reached, and those whose window wraps (p = 0) in this tensor
    std::vector<uint8_t> seen(65536, 0), seen_wrap(65536, 0);
    uint16_t st[256];
    const size_t tw = 16 * (size_t) K, tiles = l.tr.size() / tw;
    for (size_t t = 0; t < tiles; ++t)
    {
        exo_tile_states(K, &l.tr[t * tw], st);
        for (int p = 0; p < 256; ++p) seen[st[p]] = 1;
        seen_wrap[st[0]] = 1;
    }
    size_t states = 0, wrap_states = 0;
    for (int i = 0; i < 65536; ++i) { states += seen[i]; wrap_states += seen_wrap[i]; }
    ELin e = eupload(l);
    uint16_t* dw = dalloc<uint16_t>((size_t) l.in * l.out);
    const UnpackP up{(uint64_t) e.words, (uint64_t) dw, l.out, e.stride_k(), e.stride_nb()};
    launch(fn("tf_ds_exl3_unpack_k" + std::to_string(e.k2)), dim3(l.out / 16, l.in / 16), dim3(32), 0, up);
    HC(hipStreamSynchronize(g_stream));
    auto gw = download(dw, (size_t) l.in * l.out);
    std::vector<uint16_t> wq = cpu_decode(l);
    size_t diff = 0, neg = 0;
    for (size_t j = 0; j < wq.size(); ++j) { diff += wq[j] != gw[j]; neg += (wq[j] >> 15) & 1; }
    // the codebook itself over all states, through the oracle's two routes (table and per-state)
    std::vector<uint16_t> tab(65536);
    exo_mul1_table(tab.data());
    size_t cb_diff = 0;
    for (int i = 0; i < 65536; ++i) cb_diff += tab[i] != exo_mul1((uint16_t) i);
    const bool ok = diff == 0 && cb_diff == 0;
    printf("{\"mode\":\"exact\",\"tensor\":\"%s\",\"K\":%d,\"K2\":%d,\"k\":%d,\"n\":%d,\"tiles\":%zu,\"values\":%zu,\"states\":%zu,"
           "\"wrap_states\":%zu,\"negative_values\":%zu,\"mismatches\":%zu,\"codebook_route_diff\":%zu,\"kernel\":\"tf_ds_exl3_unpack_k%d\","
           "\"object\":\"%s\",\"pass\":%s}\n",
           tensor.c_str(), K, 2 * K, l.in, l.out, tiles, wq.size(), states, wrap_states, neg, diff, cb_diff, 2 * K,
           getenv("EXL3_HSACO") ? getenv("EXL3_HSACO") : "", ok ? "true" : "false");
    fflush(stdout);
    return ok ? 0 : 1;
}

// ==== k-dec: row invariance ==========================================================================================
// A row's outputs must be the same bits whatever window of a decode / verify round it shares a launch with (1 to 63
// rows): each row of a window (drawn from a pool, in random order, repeats allowed) against the same row run alone.

// a synthetic EXL3 linear: random trellis words (every 16-bit state is valid), suh +-[0.5, 1.5), svh [0.002, 0.012)
static Lin synth_lin(const std::string& name, int k, int n, int K, uint64_t seed)
{
    std::mt19937_64 r(seed);
    auto u01 = [&] { return 0x1p-53 * (double) (r() >> 11); };
    Lin l;
    l.name = name; l.K = K; l.in = k; l.out = n;
    l.tr.resize((size_t) (k / 16) * (n / 16) * 16 * K);
    for (auto& v : l.tr) v = (uint16_t) r();
    l.suh.resize(k);
    for (auto& v : l.suh) v = exo_f64_to_f16(((r() & 1) ? -1.0 : 1.0) * (0.5 + u01()));
    l.svh.resize(n);
    for (auto& v : l.svh) v = exo_f64_to_f16(0.002 + 0.01 * u01());
    return l;
}

// bit differences of a got row against its solo row: (floats differing, max |ulp| distance)
static void bitdiff(const float* got, const float* solo, size_t n, size_t& diff, int64_t& max_ulp)
{
    for (size_t j = 0; j < n; ++j)
    {
        uint32_t a, b;
        memcpy(&a, got + j, 4); memcpy(&b, solo + j, 4);
        if (a == b) continue;
        diff++;
        const int64_t ia = (a & 0x80000000u) ? -(int64_t) (a & 0x7fffffffu) : (int64_t) a;
        const int64_t ib = (b & 0x80000000u) ? -(int64_t) (b & 0x7fffffffu) : (int64_t) b;
        max_ulp = std::max<int64_t>(max_ulp, ia > ib ? ia - ib : ib - ia);
    }
}

static std::vector<int> window_sizes(int rmax)
{
    std::vector<int> v;
    for (int r : {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 31, 32, 33, 47, 48, 63})
        if (r <= rmax) v.push_back(r);
    return v;
}

// glinear groups (the real groups(), or SYNTH=1: every K2 at a (SK > 1, 4 waves) and a (SK 1, 8 waves) shape)
static int mode_rowinv_gl(int rmax, const std::string& gname)
{
    std::vector<GroupCase> gcs;
    const bool synth = env_int("SYNTH", 0) != 0;
    if (synth)
        for (int K : {2, 3, 4, 5, 6, 8}) gcs.push_back({"synth_k" + std::to_string(K), {}, 0, 0, 0});
    else gcs = groups();
    const int POOL = 24, TRIALS = env_int("TRIALS", 2), MMAX = std::min(rmax, 63);
    int rc = 0;
    for (const GroupCase& gc : gcs)
    {
        if (!group_selected(gname, gc.name)) continue;
        std::vector<Lin> L;
        if (synth)
        {
            const int K = atoi(gc.name.c_str() + 7);
            L.push_back(synth_lin(gc.name + ".a", 2048, 1024, K, 100 + K));    // plan SK 4, 4 waves
            L.push_back(synth_lin(gc.name + ".b", 1024, 8192, K, 200 + K));    // plan SK 1, 8 waves
            L.push_back(synth_lin(gc.name + ".c", 4096, 1024, K, 300 + K));    // plan SK 8, 4 waves
        }
        else for (auto& t : gc.tensors) L.push_back(make_lin(t, gc.slice, gc.a, gc.b));
        const int n = (int) L.size();
        std::vector<ELin> E;
        for (auto& l : L) E.push_back(eupload(l));
        std::vector<std::vector<uint16_t>> pool(n);                          // POOL bf16 rows a layer
        std::vector<uint16_t*> dx(n), dxh(n);
        std::vector<float*> dy(n);
        std::vector<int32_t*> cnt(n);
        for (int i = 0; i < n; ++i)
        {
            std::vector<uint16_t> h = rand_x((size_t) POOL * L[i].in);
            pool[i].resize(h.size());
            for (size_t j = 0; j < h.size(); ++j) pool[i][j] = f_to_bf((float) exo_f16_to_f64(h[j]));
            dx[i] = dalloc<uint16_t>((size_t) MMAX * L[i].in);
            dxh[i] = dalloc<uint16_t>((size_t) MMAX * L[i].in);
            dy[i] = dalloc<float>((size_t) MMAX * L[i].out);
            cnt[i] = dalloc<int32_t>((size_t) 8 * L[i].out / 128);
            dzero(cnt[i], (size_t) 8 * L[i].out / 128 * 4);
        }
        std::vector<Call> calls;
        for (int i = 0; i < n; ++i) calls.push_back({&E[i], dx[i], L[i].in, 1, dxh[i], dy[i], L[i].out, 2, cnt[i]});
        float* dz = dalloc<float>(std::max<size_t>(1, z_floats(calls, MMAX)));
        // run rows idx (pool indices) as one window: y [layer][row][N]
        auto run = [&](const std::vector<int>& idx) {
            const int M = (int) idx.size();
            for (int i = 0; i < n; ++i)
            {
                std::vector<uint16_t> xr((size_t) M * L[i].in);
                for (int r = 0; r < M; ++r) memcpy(&xr[(size_t) r * L[i].in], &pool[i][(size_t) idx[r] * L[i].in], (size_t) L[i].in * 2);
                HC(hipMemcpy(dx[i], xr.data(), xr.size() * 2, hipMemcpyHostToDevice));
            }
            HC(hipDeviceSynchronize());
            rot_many(calls, M);
            glinear(calls, M, dz);
            HC(hipStreamSynchronize(g_stream));
            std::vector<std::vector<float>> y(n);
            for (int i = 0; i < n; ++i) y[i] = download(dy[i], (size_t) M * L[i].out);
            return y;
        };
        std::vector<std::vector<std::vector<float>>> solo(POOL);
        for (int p = 0; p < POOL; ++p) solo[p] = run({p});
        std::mt19937_64 rr(77);
        std::string per;
        size_t total_rows = 0, total_diff = 0;
        int64_t total_ulp = 0;
        for (int R : window_sizes(MMAX))
        {
            size_t diff = 0, rows = 0;
            int64_t mu = 0;
            for (int t = 0; t < TRIALS; ++t)
            {
                std::vector<int> idx(R);
                for (int r = 0; r < R; ++r) idx[r] = (int) (rr() % POOL);
                auto y = run(idx);
                for (int i = 0; i < n; ++i)
                    for (int r = 0; r < R; ++r) bitdiff(&y[i][(size_t) r * L[i].out], solo[idx[r]][i].data(), L[i].out, diff, mu);
                rows += R;
            }
            char buf[96];
            snprintf(buf, sizeof buf, "%s{\"R\":%d,\"rows\":%zu,\"diff\":%zu,\"max_ulp\":%lld}", per.empty() ? "" : ",", R, rows, diff, (long long) mu);
            per += buf;
            total_rows += rows; total_diff += diff; total_ulp = std::max(total_ulp, mu);
        }
        size_t nonzero = 0;
        for (int i = 0; i < n; ++i) { auto cv = download(cnt[i], (size_t) 8 * L[i].out / 128); for (int v : cv) nonzero += v != 0; }
        std::string shapes;
        for (int i = 0; i < n; ++i)
        {
            const Plan p = plan(L[i].in, L[i].out);
            char buf[160];
            snprintf(buf, sizeof buf, "%s{\"tensor\":\"%s\",\"K\":%d,\"k\":%d,\"n\":%d,\"sk\":%d,\"wk\":%d}", i ? "," : "", L[i].name.c_str(), L[i].K, L[i].in, L[i].out, p.sk, p.wk);
            shapes += buf;
        }
        const bool ok = total_diff == 0 && nonzero == 0;
        printf("{\"mode\":\"rowinv_gl\",\"group\":\"%s\",\"layers\":[%s],\"pool\":%d,\"window_rows\":%zu,\"diff\":%zu,\"max_ulp\":%lld,\"counters_nonzero\":%zu,\"by_R\":[%s],\"pass\":%s}\n",
               gc.name.c_str(), shapes.c_str(), POOL, total_rows, total_diff, (long long) total_ulp, nonzero, per.c_str(), ok ? "true" : "false");
        fflush(stdout);
        rc |= ok ? 0 : 1;
        for (void* p : g_allocs) HC(hipFree(p));
        g_allocs.clear();
    }
    return rc;
}

// a synthetic expert table: `routed` experts and the shared one, K2 cycling through 2, 4, .. 16 (K 1-8) per expert
// and matrix (the lo2_hi16 range), D 5120, I 1152 (a TP2 rank's)
static EExperts synth_eexperts(int routed)
{
    EExperts x;
    x.E = routed + 1;
    const int D = 5120, I = 1152;
    for (int e = 0; e < x.E; ++e)
    {
        x.g.push_back(synth_lin("synth.g" + std::to_string(e), D, I, 1 + (e % 8), 1000 + e));
        x.u.push_back(synth_lin("synth.u" + std::to_string(e), D, I, 1 + ((e + 3) % 8), 2000 + e));
        x.d.push_back(synth_lin("synth.d" + std::to_string(e), I, D, 1 + ((e + 5) % 8), 3000 + e));
    }
    x.D = D;
    x.I = I;
    std::vector<int64_t> gp, upp, dp;
    std::vector<int32_t> gk, uk, dk;
    std::vector<uint16_t> sg, su, vg, vu, sd, vd;
    for (int e = 0; e < x.E; ++e)
    {
        gp.push_back((int64_t) dupload(x.g[e].tr)); upp.push_back((int64_t) dupload(x.u[e].tr)); dp.push_back((int64_t) dupload(x.d[e].tr));
        gk.push_back(2 * x.g[e].K); uk.push_back(2 * x.u[e].K); dk.push_back(2 * x.d[e].K);
        sg.insert(sg.end(), x.g[e].suh.begin(), x.g[e].suh.end()); su.insert(su.end(), x.u[e].suh.begin(), x.u[e].suh.end());
        vg.insert(vg.end(), x.g[e].svh.begin(), x.g[e].svh.end()); vu.insert(vu.end(), x.u[e].svh.begin(), x.u[e].svh.end());
        sd.insert(sd.end(), x.d[e].suh.begin(), x.d[e].suh.end()); vd.insert(vd.end(), x.d[e].svh.begin(), x.d[e].svh.end());
        for (int k2 : {gk.back(), uk.back()}) { x.k2_gu[0] = std::min(x.k2_gu[0], k2); x.k2_gu[1] = std::max(x.k2_gu[1], k2); }
        x.k2_d[0] = std::min(x.k2_d[0], dk.back()); x.k2_d[1] = std::max(x.k2_d[1], dk.back());
    }
    x.gate_ptr = dupload(gp); x.up_ptr = dupload(upp); x.down_ptr = dupload(dp);
    x.gate_k2 = dupload(gk); x.up_k2 = dupload(uk); x.down_k2 = dupload(dk);
    x.suh_g = dupload(sg); x.suh_u = dupload(su); x.svh_g = dupload(vg); x.svh_u = dupload(vu); x.suh_d = dupload(sd); x.svh_d = dupload(vd);
    return x;
}

// decode experts: pool rows routed to a hot set of HOT experts (rows 0 .. POOL/2 - 1: many experts shared, 1 to R
// members) or anywhere (the rest: mostly lone members); the shared expert last in every row. Windows: all hot, all
// cold, mixed. LAYER -1: the synthetic table (SYNTH_ROUTED experts).
static int mode_rowinv_x(int rmax, int layer)
{
    const int slots = env_int("SLOTS", 7), POOL = 32, HOT = env_int("HOT", 10), TRIALS = env_int("TRIALS", 3);
    const int routed = layer < 0 ? env_int("SYNTH_ROUTED", 64) : env_int("ROUTED", 384);
    const float limit = 10.0f;
    EExperts x = layer < 0 ? synth_eexperts(routed) : load_eexperts(layer, routed);
    const int D = x.D, MMAX = std::min(rmax, 63);
    DecodeScratch sc = alloc_decode(64, slots, D, x.I, x.E);
    std::vector<uint16_t> h = rand_x((size_t) POOL * D), pool(h.size());
    for (size_t j = 0; j < h.size(); ++j) pool[j] = f_to_bf((float) exo_f16_to_f64(h[j]));
    std::mt19937_64 rng(4242);
    std::vector<int> hot(routed);
    std::iota(hot.begin(), hot.end(), 0);
    std::shuffle(hot.begin(), hot.end(), rng);
    hot.resize(HOT);
    std::vector<int32_t> ppick((size_t) POOL * slots);
    std::vector<float> pwts(ppick.size());
    for (int r = 0; r < POOL; ++r)
    {
        std::vector<int> cand = r < POOL / 2 ? hot : std::vector<int>();
        if (cand.empty()) { cand.resize(routed); std::iota(cand.begin(), cand.end(), 0); }
        std::shuffle(cand.begin(), cand.end(), rng);
        for (int s = 0; s < slots - 1; ++s)
        {
            ppick[(size_t) r * slots + s] = cand[s];
            pwts[(size_t) r * slots + s] = 0.05f + 0.25f * (float) ((rng() >> 11) * 0x1p-53);
        }
        ppick[(size_t) r * slots + slots - 1] = routed;
        pwts[(size_t) r * slots + slots - 1] = 1.0f;
    }
    uint16_t* dx = dalloc<uint16_t>((size_t) MMAX * D);
    int32_t* dpick = dalloc<int32_t>((size_t) MMAX * slots);
    float* dwts = dalloc<float>((size_t) MMAX * slots);
    float* dout = dalloc<float>((size_t) MMAX * D);
    int max_members = 0;
    auto run = [&](const std::vector<int>& idx) {
        const int R = (int) idx.size();
        std::vector<uint16_t> xr((size_t) R * D);
        std::vector<int32_t> pk((size_t) R * slots);
        std::vector<float> wt(pk.size());
        std::map<int, int> members;
        for (int r = 0; r < R; ++r)
        {
            memcpy(&xr[(size_t) r * D], &pool[(size_t) idx[r] * D], (size_t) D * 2);
            memcpy(&pk[(size_t) r * slots], &ppick[(size_t) idx[r] * slots], (size_t) slots * 4);
            memcpy(&wt[(size_t) r * slots], &pwts[(size_t) idx[r] * slots], (size_t) slots * 4);
            for (int s = 0; s < slots; ++s) max_members = std::max(max_members, ++members[pk[(size_t) r * slots + s]]);
        }
        HC(hipMemcpy(dx, xr.data(), xr.size() * 2, hipMemcpyHostToDevice));
        HC(hipMemcpy(dpick, pk.data(), pk.size() * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dwts, wt.data(), wt.size() * 4, hipMemcpyHostToDevice));
        HC(hipDeviceSynchronize());
        decode_experts(x, sc, dx, dpick, dwts, dout, R, limit);
        HC(hipStreamSynchronize(g_stream));
        return download(dout, (size_t) R * D);
    };
    std::vector<std::vector<float>> solo(POOL);
    for (int p = 0; p < POOL; ++p) solo[p] = run({p});
    std::mt19937_64 rr(99);
    std::string per;
    size_t total_rows = 0, total_diff = 0;
    int64_t total_ulp = 0;
    for (int R : window_sizes(MMAX))
    {
        size_t diff = 0, rows = 0;
        int64_t mu = 0;
        for (int t = 0; t < TRIALS; ++t)
        {
            // trial 0: hot rows only, 1: cold rows only, 2..: anywhere (repeats when R exceeds the half)
            std::vector<int> idx(R);
            for (int r = 0; r < R; ++r)
                idx[r] = t == 0 ? (int) (rr() % (POOL / 2)) : t == 1 ? POOL / 2 + (int) (rr() % (POOL / 2)) : (int) (rr() % POOL);
            auto y = run(idx);
            for (int r = 0; r < R; ++r) bitdiff(&y[(size_t) r * D], solo[idx[r]].data(), D, diff, mu);
            rows += R;
        }
        char buf[96];
        snprintf(buf, sizeof buf, "%s{\"R\":%d,\"rows\":%zu,\"diff\":%zu,\"max_ulp\":%lld}", per.empty() ? "" : ",", R, rows, diff, (long long) mu);
        per += buf;
        total_rows += rows; total_diff += diff; total_ulp = std::max(total_ulp, mu);
    }
    // the scratch protocol after every call: counters zero, the used places ready = epoch
    const int ep = download(sc.epoch, 1)[0];
    const int nu = download(sc.count, 1)[0];
    auto rdy = download(sc.ready, std::max(1, sc.maxu));
    size_t ready_bad = 0, cnt_nonzero = 0;
    for (int u = 0; u < nu; ++u) ready_bad += rdy[u] != ep;
    for (int v : download(sc.ready_cnt, std::max(1, sc.maxu))) cnt_nonzero += v != 0;
    for (int v : download(sc.cnt_gu, (size_t) sc.maxu * (x.I / 128))) cnt_nonzero += v != 0;
    for (int v : download(sc.cnt_d, (size_t) sc.rows * (D / 128))) cnt_nonzero += v != 0;
    const bool ok = total_diff == 0 && ready_bad == 0 && cnt_nonzero == 0;
    printf("{\"mode\":\"rowinv_x\",\"layer\":%d,\"mtp\":%s,\"experts\":%d,\"slots\":%d,\"k2_gu\":[%d,%d],\"k2_d\":[%d,%d],\"hot\":%d,\"pool\":%d,"
           "\"max_members\":%d,\"window_rows\":%zu,\"diff\":%zu,\"max_ulp\":%lld,\"epoch\":%d,\"ready_bad\":%zu,\"counters_nonzero\":%zu,\"by_R\":[%s],\"pass\":%s}\n",
           layer, getenv("MTP") ? "true" : "false", x.E, slots, x.k2_gu[0], x.k2_gu[1], x.k2_d[0], x.k2_d[1], HOT, POOL, max_members, total_rows,
           total_diff, (long long) total_ulp, ep, ready_bad, cnt_nonzero, per.c_str(), ok ? "true" : "false");
    fflush(stdout);
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    ROOT = std::string(getenv("HOME")) + "/tfp/exl3/";
    if (argc < 3) DIE("usage: eng glinear R GROUP | prefill M TENSOR | dexperts R LAYER | pexperts M LAYER");
    load_manifest();
    HC(hipSetDevice(0));
    HC(hipStreamCreateWithFlags(&g_stream, hipStreamNonBlocking));
    const char* obj = getenv("EXL3_HSACO");
    const std::string path = obj ? obj : ROOT + "tmp/kit/exl3.hsaco";
    HC(hipModuleLoad(&g_mod, path.c_str()));
    const std::string m = argv[1];
    const std::string arg = argc > 3 ? argv[3] : "all";
    int rc = 0;
    // R may be a comma-separated list: the mode once for each (one process, one GPU lock)
    std::vector<int> rs;
    for (const char* q = argv[2]; *q;) { rs.push_back(atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
    for (const int R : rs) {
    if (m == "glinear") rc |= mode_glinear(R, arg);
    else if (m == "prefill") rc |= mode_prefill(R, arg);
    else if (m == "dexperts") rc |= mode_dexperts(R, argc > 3 ? atoi(argv[3]) : 39);
    else if (m == "pexperts") rc |= mode_pexperts(R, argc > 3 ? atoi(argv[3]) : 39);
    else if (m == "pfexperts") rc |= mode_pfexperts(R, argc > 3 ? atoi(argv[3]) : 39);
    else if (m == "pgemm") rc |= mode_pgemm(R);
    else if (m == "rowinv_gl") rc |= mode_rowinv_gl(R, arg);
    else if (m == "exact") rc |= mode_exact(R, arg);
    else if (m == "l2pf") rc |= mode_l2pf(R, arg);
    else if (m == "rowinv_x") rc |= mode_rowinv_x(R, argc > 3 ? atoi(argv[3]) : 39);
    else DIE("unknown mode %s", m.c_str());
    HC(hipDeviceSynchronize());
    for (void* p : g_allocs) HC(hipFree(p));   // each R's tensors and scratch, before the next
    g_allocs.clear();
    }
    HC(hipDeviceSynchronize());
    for (void* p : g_allocs) HC(hipFree(p));
    return rc;
}
