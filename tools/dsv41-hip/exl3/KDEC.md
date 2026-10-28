# k-dec: EXL3 decode kernels (row invariance, then speed)

Owner: k-dec (2026-10-19). Branch `kdec` (node1 ~/tfp/wt/kdec), work area node1 ~/tfp/kdec (dev objects, harness,
results, logs). Kernels: zig/kernels/hip/dsv41/exl3.hip (glinear, grouped_cp).

## Tools (kdec/ here; they run from node1 ~/tfp/kdec)
Setup: `mkdir -p ~/tfp/kdec/{hip,tools,tmp,src,logs,results,conf,runs}`, `cp kdec/harness.hip ~/tfp/kdec/hip/kdec.hip`,
`cp kdec/mb.hip kdec/bw.hip ~/tfp/kdec/hip/`, `cp kdec/*.sh kdec/*.py ~/tfp/kdec/tools/`. The harness needs w-exl3's
node1 ~/tfp/exl3 (hip/hostcommon.h, oracle/libexl3_oracle.so, the tensor manifest over the checkpoint).
- `harness.hip`: w-exl3's engine-ABI harness (~/tfp/exl3/hip/eng.hip: kernels loaded from a code object and launched
  as exl3_linear.zig / exl3_experts.zig pack them, real tensors, w-ref's contract; modes glinear, dexperts, prefill, ...)
  plus: R may be a list (`glinear 1,2,3 all`: one process, one GPU lock); HIP-graph replay timing (`graph_us`); SK_GU
  (gate / up's splits), LEAN / SPLIT (the lean entry points of the patch below); and two modes:
  - `rowinv_gl RMAX GROUP` (SYNTH=1: synthetic tensors, every K2 at three plans): every row of windows of R = 2..RMAX
    rows (drawn from a 24-row pool, random order, repeats allowed; two windows a size) against the same row run alone
    (M = 1): fp32 outputs bit for bit, split-K counters left zero.
  - `rowinv_x RMAX LAYER` (LAYER -1: a synthetic 65-expert table with K2 2..16; MTP=1 ROUTED=128: the drafter blocks):
    pool rows routed into a hot set of 10 experts (experts shared by 1..R rows) or anywhere (mostly lone members), the
    shared expert in every row; windows all hot / all cold / mixed; every row's final output against the row run alone
    (R = 1): bits, plus the decode scratch protocol (epoch, ready, counters).
- `mkdev.py SRC OUT`: a decode-only copy of exl3.hip (no exl3.gemm / prompt grouped instantiations; the decode kernels'
  source unchanged) that compiles in about a minute; `build_dev.sh TAG SRC` builds it with the kit's flags into
  tmp/TAG.hsaco with its resource usage; `mkexp.py SRC OUT EXP` writes the timing experiments below (never committed).
- `gpu.sh` / `runq.sh NAME CMD`: one run under the box's GPU lock (retried after a 30-minute wait) and a quiet window;
  `~/tfp/kdec/PAUSE` holds new runs (a two-box served run needs both locks); `sum.py`, `cmp.py A B` (base vs new per
  group / layer / R, with totals); `bg.sh NAME CMD`: CPU work as a run-kdec-NAME user service (upper cores, nice, caps).
- `mkconf.sh` / `serve_run.sh NAME DEPLOY KEY=VALUE...`: a served run on both boxes from DEPLOY (after a-arch's), rank
  0's `done chatcmpl` lines (tokens, SHA) and the bench in ~/tfp/kdec/runs/NAME.log. Never under bg.sh (the ranks must
  not inherit its caps).
- `mb.hip`: issue cost of single VALU instructions (8 independent chains a lane). `bw.hip`: streaming read bandwidth
  against load width and loads in flight.

## Log
### 1. Row invariance (2026-10-19)
Cause: glinear ran a pass of one row on the 1-row v_dot2 path and passes of 2-16 rows on v_wmma 16x16x16 tiles;
grouped_cp chose the same per expert tile by its member count. WMMA's fp32 sums run in another order (and are not
exact), so a row's outputs depended on the window's width and on who shared its expert.

Change: every row of a decode / verify window (glinear below EXL3_WIDE_MIN = 64 rows, grouped_cp at any member count)
runs the 1-row path's arithmetic: its own v_dot2_f32_f16 chain over the wave's k range in k order, then the two
half-waves' sums added (own half + partner), then the waves in wave order and the splits in split order, exactly as
before for one row. `strip_tiles_rows` decodes each tile once per window for all its rows; a k step's input rows are
staged in LDS (32 x 16 bytes, aliased on the reduction buffer, which is free until the pass's barrier) and read back
as broadcasts. Up to 8 rows: all 8 tiles of the strip at once (64 accumulators); 9-16 rows: two parts of 4 tiles (each
part loads only its tiles' words). The WMMA decode strip (`strip_tiles`) had no caller left and was removed; the
prompt paths (glinear_wide from 64 rows, grouped_mma / grouped_rows, exl3.gemm) keep WMMA.
Registers: glinear 197 -> 231 VGPRs (7 -> 6 waves a SIMD), grouped_cp lo2_hi10 162 -> 211 (6 waves either way: LDS
bound), no spills, no scratch (a first version spilled: 16 x 8 accumulators; one kept the C vectors in scratch
through an element-wise loop with an early exit).

Kernel test (node1, dev object new4 = this source minus the prompt kernels' instantiations; base = strix-port 86302cf):
| test | new4 | base |
|---|---|---|
| rowinv_gl SYNTH=1 (K 2,3,4,5,6,8; plans SK 4 / 1 / 8, 4 / 8 / 4 waves), R 2..63, 896 window rows a K | 0 floats differ | ~8.0M of 8.0M floats differ a K |
| rowinv_gl real groups qa_kv, qb, woa, wob, idx, shared, shared_w2, head | 0 differ (896 window rows each) | - |
| rowinv_x L39 (K2 6/10), 1,344 window rows, up to 63 members a place | 0 differ; epoch/ready/counters clean | 6.86M floats differ |
| rowinv_x L20 (K2 4/8) | 0 differ | - |
| rowinv_x MTP0 (128 routed + shared, lo2_hi16) | 0 differ | - |
| rowinv_x synthetic 65 experts, K2 2..16 per matrix | 0 differ | - |

Instruction costs on gfx1151 (hip/mb.hip microbenchmark, 8 independent chains a lane, issue cost relative to
v_add_nc_u32, two passes): v_mul_lo_u32 3.34 / 3.44, v_mul_hi_u32 3.31 / 3.37, v_dot4_u32_u8 1.85 / 1.89, v_mad_u32_u16
1.14 / 1.18, v_dot2_f32_f16 1.14 / 1.15, v_perm_b32 1.16 / 1.15, v_alignbit_b32 1.11 / 1.13, v_sad_u8 1.06 / 1.07,
v_pk_fma_f16 1.06 / 1.08, v_lshl_add_u32 1.05 / 1.04, v_pk_mul_lo_u16 1.05 / 1.04, v_mul_u32_u24 0.97 / 0.99,
v_lshrrev_b32 0.92 / 0.94, v_bfe_u32 0.92 / 0.92. The MUL1 hash's two v_mul_lo_u32 are about half of a value pair's
VALU issue (6.8 of ~13 slots: alignbit, 16-bit move, bfe, 2 mul_lo, 2 sad, pk_fma). (The sclk this tool prints is
read after the runs and is not the clock they ran at.)

Served baseline before the change (a-arch's runs on deploy-arch, rank 0 log `done chatcmpl ... sha=`): drafts off code
071be2ea3b48 (400 tokens), prose dd15d40274f8 (393 tokens), identical in every drafts-off run and with drafts on at
FIX_K=0 (R = 1 every round); drafts on with any wider round: code 071be2ea3b48 / 7e82cb1db92a / d04fb009d9ae /
e3633f32c2fe / ..., prose a different SHA in every run (4084dd11177c, 07851cb4ccc7, 4df20f8a5314, ...).


w-ref contract (node1 harness modes `glinear R all`, `dexperts R L`; dev object new5 = the committed source minus the
prompt kernels' instantiations), R = 1..6: glinear 48 records pass (bound ratio <= 0.0009, NRMSE <= 2.22e-4, 0
probabilistic exceedances, bf16 y / folded rotation / RoPE bit-exact against the CPU emulation, split-K counters zero);
decode experts L39, L20, MTP0 pass at every R (NRMSE <= 3.94e-4, worst row <= 4.30e-4; epoch / ready / counters
clean).

Verify-window cost, R = 1..6 (pass 1: base then new5, each run one process under the GPU lock and a quiet window;
medians and back-to-back means of 100 cold-weight iterations; sclk ~2.0-2.5 GHz on the dense groups, ~1.3 on the head,
1.0-1.2 GHz on the experts; ratio new5 / base):
| R | glinear 8 groups, sum (us) | ratio med / b2b | experts L39 (us) | ratio | L20 | ratio | MTP0 | ratio |
|---|---|---|---|---|---|---|---|---|
| 1 | 1656.6 -> 1640.5 | 0.990 / 0.994 | 269.6 -> 274.0 | 1.016 / 1.002 | 233.8 -> 233.9 | 1.000 / 1.004 | 329.2 -> 333.4 | 1.013 / 1.004 |
| 2 | 1681.3 -> 1639.4 | 0.975 / 0.974 | 460.4 -> 464.9 | 1.010 / 1.001 | 405.5 -> 404.0 | 0.996 / 1.009 | 569.3 -> 572.6 | 1.006 / 1.001 |
| 3 | 1685.4 -> 1647.9 | 0.978 / 0.977 | 641.9 -> 642.2 | 1.000 / 0.984 | 575.8 -> 564.2 | 0.980 / 0.984 | 772.4 -> 781.7 | 1.012 / 1.002 |
| 4 | 1681.6 -> 1656.7 | 0.985 / 0.982 | 821.8 -> 820.5 | 0.998 / 0.993 | 746.3 -> 727.0 | 0.974 / 0.985 | 965.7 -> 975.9 | 1.011 / 1.003 |
| 5 | 1691.4 -> 1689.2 | 0.999 / 0.987 | 999.5 -> 993.3 | 0.994 / 0.991 | 903.9 -> 903.8 | 1.000 / 1.000 | 1173.4 -> 1174.4 | 1.001 / 1.001 |
| 6 | 1688.2 -> 1697.5 | 1.006 / 1.004 | 1165.9 -> 1152.2 | 0.988 / 0.985 | 1058.3 -> 1074.5 | 1.015 / 1.111 | 1352.9 -> 1360.3 | 1.005 / 1.000 |
Per group the dense ratios spread 0.82-1.24 (small groups such as idx at 43 us are noisy; shared runs +3..+14% at R
4-6, shared_w2 -10..-16% at every R). The experts' time follows the places (unique experts) almost linearly, so a
window's cost is set by its routed experts' weight reads; the row-invariant arithmetic itself costs nothing measurable.
Second pass (reverse order), both passes' means, new5 / base, median and back-to-back over R = 1..6: glinear 0.975-1.003
/ 0.973-1.000; L39 1.000-1.004 / 0.972-1.004; L20 0.984-1.006 / 0.985-1.054 (the R = 6 back-to-back outlier again);
MTP0 0.997-1.030 / 0.997-1.003.

Served acceptance (deploy-kdec/840f280eeb50 on both boxes; serve.conf defaults: tcp_proxy + HIP graphs, GB10 round
profile; GPU_STRICT=1; code and prose, 400 tokens; ~/tfp/run/serve/kdec-kd1-*; rank 0's `done chatcmpl` lines):
| run | code sha | code tok/s | prose sha | prose tok/s | code accepted / drafted |
|---|---|---|---|---|---|
| kd1-off (drafts off) | 48951eea1fc4 | 32.19 | 62a2c277a61d | 31.91 | - |
| kd1-on1 (drafts on) | 48951eea1fc4 | 58.43 | 62a2c277a61d | 39.95 | 308 / 407 |
| kd1-on2 (drafts on) | 48951eea1fc4 | 69.44 | 62a2c277a61d | 40.85 | 308 / 393 |
Drafts-on greedy equals drafts-off and reproduces across runs although the windows' widths differ between the runs (the
drafted counts). Merged into strix-port as a11df7c (`current` on both boxes since ~18:20). The drafts-off SHAs differ from strix-port's (071be2ea3b48 / dd15d40274f8): the 31- and 34-token
prompts run glinear and grouped_cp below 64 rows, which took WMMA before. Speeds are not an interleaved comparison
(a-arch's runs on 6a0c789 with the same settings: drafts off 30.4 / 28.7-29.8, drafts on 52.6-63.1 / 33.9-37.1).

### 2. Faster 1-row and small-R decode
Where the time goes (a-arch's rank-1 trace, drafts on, node1 ~/tfp/arch/aarch/prof/dr-prof2): grouped_cp lo2_hi10 epi1
28.4% (mean 429 us a launch) and epi2 15.8% (239 us), tf_ds_coll_collect 14.4%, glinear k10_w4 12.7% (61.8 us), k10_w8
5.7%, k12_w8 4.4%, k12_w4 2.0%; everything else below 2.1% (rot_many 0.3%, decode_prep 0.6%).

Harness numbers now (new5 = the row-invariant source; cold weights; HIP-graph replay of 32 launch groups, which equals
back-to-back timing within 1-2%):
- Decode experts, a place's (unique expert's) marginal cost from the R = 1..6 scan: L20 (K2 4) 28.9 us, L39 (K2 6) 31.3
  us, MTP0 (K2 8) 38.8 us, for the same 17.7M values a place (gate, up, down of 5120 x 1152): 153 / 211 / 227 GB/s. The
  2-bit layers are compute-bound (612 G values/s at sclk ~1.1 GHz under DPM auto), the 4-bit ones memory-bound, the
  3-bit ones (35 of 40 routed layers) near both limits. Fixed cost a layer at R = 1: gate / up ~20 us, down ~11 us,
  decode_prep ~4 us (intercepts of the scan); L39 R = 1 264 us (193 GB/s), R = 6 1140 us (211 GB/s).
- Dense groups at R = 1: head 202 GB/s (248 MB, one launch); the small groups 119-183 GB/s (qa_kv 5.7 MB 37 us, idx 5.0
  MB 42 us, woa 10.5 MB 63 us, shared 13.3 MB 85 us): a launch's fixed cost is ~8-10 us at 5-15 MB.
- sclk sampled during the runs: decode experts 1.0-1.2 GHz, dense groups 1.3-2.5 GHz, head 1.2-1.4 GHz.

Tried, measured, not kept:
| idea | result |
|---|---|
| Lean single-row entry points (glinear1 / grouped_cp1: the 1-row path alone; 71-90 VGPRs, 7.5 KB LDS, 16 waves a SIMD vs 6) and a split launch for wider windows (grouped_cp1 for one-member places + grouped_cpm for the rest) | only1 experiment at R = 1, two passes vs new5: L39 +4.7% slower, MTP0 +4.9% / +8.2% (median / b2b), L20 -3.3%, glinear groups -0.6%. More waves did not help: the experts are not latency-bound at 6 waves a SIMD. glinear1 is bit-identical to glinear (rowinv, synthetic, 0 diffs) but its 1% is noise. Patch kept in node1 ~/tfp/kdec/src/d2_lean_split.patch. |
| MUL1 hash without v_mul_lo_u32 (pkhash: one v_pk_mul_lo_u16 for both values' high products, v_mad_u32_u16 for the low products and sums; 7.4 issue slots a pair for hash + extraction vs 8.7 by the microbenchmark) | Decode exact (unpack bit for bit against the oracle on 6 real tensors, K 2-6 and 8), but slower: experts L39 +11..+23%, L20 +21..+26%, MTP0 +4..+5%, glinear -1%. The dependent chain costs more than the issue count says. |

| gate / up K splits (SK_GU, the plan's SK = 4; harness override, numerics change but stay invariant and in contract) | graph timing L39 R = 1 / 6: SK 2 267 / 1129 us, 4 263 / 1140, 8 275 / 1207, 16 294 / 1336; L20: 230 / 1028, 228 / 1041, 235 / 1089, 262 / 1212. SK 2 and 4 within noise, more splits slower: the launch's fixed cost is not a tail of too few programs. Plan left as is. |
| 1-row strip with two k steps of words in flight (pd2; the same arithmetic) | one pass vs new5 (graph): glinear -0.8% (R 1) .. +0.1%, experts L39 -0.3..+1.1%, L20 +0.5..+2.1%: no gain, so a wave is not waiting on its next step's words. pd3 not run. |

Streaming read ceiling (hip/bw.hip: each wave reads contiguous chunks of a 1 GiB buffer; node1, quiet window, load 2.9):
4-byte coalesced loads, 6-12 in flight: 227-234 GB/s on 6 KB chunks, 209-224 on 48 KB; 16-byte loads, 2 in flight:
231-239 on 6 KB, 179-192 on 48 KB (4 in flight: 143-222). So ~230 GB/s is what a read stream gets here: the decode
experts reach 84% of it at R = 1 (193 GB/s, L39) and 92% at R = 6 (211), the head 88% (202), the small dense groups
52-80%.

Ceiling without the decode (nodecode: the MUL1 decode replaced by one AND; wrong numbers, timing only; graph timing vs
new5): glinear groups -1.4% (R 1) / -2.5% (R 6); experts L39 -6.0% / -7.2%, L20 -19.1% / -25.2%, MTP0 -3.4% / -1.1%.
Without the decode L39 R = 6 runs at 228 GB/s, the streaming ceiling. So the decode's ALU costs the 3-bit experts about
7%, the 2-bit ones (layers 18-22) about 20-25%, the dense groups almost nothing: the dense groups' shortfall is the
launches' structure (ramp, the split-K epilogue, tail), not the decode.
| 16-byte weight loads in the 1-row strip (ld128: a k step's words as b128 loads and b128 LDS stores, same arithmetic) | one pass vs new5 (graph): glinear -0.4..+0.3%, experts L39 -1.1..-0.2%, L20 -0.8..+1.0%, MTP0 -1.0..-0.2%: noise. |
| glinear's plan (blocks target 96-384 x min k tiles a wave 4 / 8 / 16; exl3_linear.zig plan is 192 / 8; harness override; SK / WK change the numerics, every row alike; all contracts pass) | graph timing, sum of the 7 groups without the head, default 408.0 us (R 1) / 454.1 (R 6); best single setting 192_t4 / 288_t8: 397.6 / 452.6 (-2.5% / -0.3%). Per shape the best SK differs: wq_a + wkv (5120 -> 1280, 512) SK 8 -> 4: 36.5 -> 32.6 us (R 1), 43.8 -> 37.5 (R 6), -11% / -14%; the idx group's (1280 -> 4096 SK 2 -> 1, 5120 -> 512 8 -> 4): 41.4 -> 37.9, 48.1 -> 43.9 (-8% / -9%); shared-expert shapes 5120 -> 2304 8 -> 4 (-4% / -9%), 2304 -> 5120 4 -> 2 (-13% / -12%); wo_a (4096 -> 1024 x 4) and wo_b (4096 -> 5120) are best at the default SK 8 at R 6 (fewer splits +23..+34%, more splits mixed); wq_b (1280 -> 16384, 8 waves, SK 1) and the head best at the default. Not applied: a per-shape table would save ~8 us a layer (~1% of a round), below what served runs resolve. |
| w-exl3's EXL3_MUL24 hash (two v_mul_u32_u24 and a shift-add a value instead of v_mul_lo_u32) | one pass vs new5 (graph), R 1..6: L39 +0.3..+1.2%, L20 -1.0..+0.2%, MTP0 +0.3..+1.0%: noise, even where the decode costs 20-25% (L20). The multiply is not what the decode's time is spent on. |

Wider windows and short prompts (R = 8..63: glinear in 16-row passes, grouped_cp with up to 63 members a place; one
pass, new5 vs 86302cf's WMMA, graph timing): the 8 glinear groups summed 1.000 at R 8, then 1.305 (R 16), 1.274 (32),
1.246 (48), 1.223 (63): above 8 rows a row's own dot2 chain costs more than a shared 16 x 16 WMMA tile (head +31% at
R 16, wq_b +37%, wo_a +18%). Experts L39 0.980 (R 8), 0.967 (16), 0.946 (32), 0.936 (48), 0.932 (63): faster, because
most places have 1-3 members. This matters for multi-stream rounds (up to ROUND_ROWS = 16 rows) and prompts of 17-63
tokens; the served single stream (k_max 5: R <= 6) is not affected. Prompt chunks could keep WMMA, since prefill does
not need to match decode windows, but only if no round ever mixes prompt rows with other rows. That is an engine
question, left open.

Conclusion of item 2 (2026-10-19): no change kept. The decode kernels run at 84% (R 1) to 92% (R 6) of the 230 GB/s
read ceiling on the 3-bit experts, 93-99% on the 4-bit ones; the decode's ALU costs ~7% there and ~20-25% on the
2-bit layers 18-22 at the ~1.1 GHz DPM auto gives these kernels (power_dpm_force_performance_level, not tested: not
mine to change). The dense groups lose 20-48% to their launches' fixed costs at 5-15 MB; a per-shape SK table would
recover ~8 us a layer. Larger steps would be structural:
- a gate / up + down launch without the ready hand-off;
- persistent decode experts;
- a contiguous-row expert layout;
- folding rot_many into the producers.

### 3. Decode time budget on the production schedule (k-dec2, 2026-10-20)
Runs (node1 unit run-kdec2-q20, tools q20.sh / budget_run.sh): deploy b010ee4ae317 on both boxes, its serve.conf (tcp_proxy +
HIP graphs, 1 stream, CONTEXT 8192, amd-1 profile), code + prose prompts, 400 tokens each; both ranks under
`rocprofv3 --kernel-trace --memory-copy-trace` (WRAP0 / WRAP1), GPU clock (hwmon freq1_input) every 1 ms and PPT every 50 ms on
both boxes during the bench. Traces: ~/tfp/kdec/prof/bud-p-{off,on}/<box>/ on each box; analysis budget.py (here): a round starts
at its _embed_init (R = grid x / 128), 83 gathers a main round (40 x attn + moe, Engram L1 / L14, head), the drafter after the
head's gather; glinear launches matched to the checkpoint's tensor groups per TP2 rank (every launch matched), expert bytes at
R = 1 (6 routed + the shared, each rank's half); the first 3 decode rounds of a request skipped. Output: prof/out-p-{off,on}.txt /
.json, per-site tables prof/sites-{off,on}.csv. clockcal.hip: rocprofv3's timestamps are CLOCK_BOOTTIME / MONOTONIC.

The profiler costs ~5%: drafts off 32.46 / 32.54 tok/s untraced vs 30.88 / 30.87 traced (code / prose), drafts on 72.65 / 41.42
vs 68.73 / 39.01; output SHAs identical in all four runs (48951eea1fc4, 62a2c277a61d).

Drafts off, per token (traced 32.38 ms; untraced 30.75 ms), rank 1 (rank 0 within 1%):
| part | ms | % | note |
|---|---|---|---|
| experts (grouped_cp) | 10.01 | 30.9 | 1,926 MB, 193 GB/s (84% of 230); ideal 8.38, excess 1.63 |
| dense glinear | 11.31 | 34.9 | 2,063 MB, 182 GB/s; ideal 8.97, excess 2.34 (wq_b 186, wo_b 193, wo_a 174, wq_a + wkv 161, head 202 GB/s) |
| router (_rowmm_parts) | 0.77 | 2.4 | 157 MB f16 |
| decode_prep | 0.17 | 0.5 | |
| small kernels | 4.10 | 12.7 | attention, norms, mixes, rope, sampling |
| gather stage + collect | 0.40 + 2.81 | 9.9 | collect: wait above each site's p5 1.39 ms, floors ~1.4 ms |
| launch gaps < 20 us | 2.20 | 6.8 | ~1,150 launches a token at ~1.9 us (graph replay; part of the profiler's 5% may be here) |
| host gaps (stretch boundaries) + idle to the next round | 0.27 + 0.32 | 1.8 | |

GPU busy (union of kernels) 29.6 of 32.4 ms. sclk median 2.85-2.89 GHz for every kernel kind, the experts too (the harness's
1.0-1.2 GHz came from its back-to-back loops, not serving); PPT median 100 W, p90 114 W. DRAM is idle between weight-streaming
launches 9.3 ms a token: experts -> the next layer's qa group 79 us x 37, wo_b -> router 70 us x 40, wq_b -> wo_a 38 us x 37,
router -> experts 28 us x 40.

Drafts on, per round (55.11 ms; code 4.5 tokens a round, prose 2.0): experts 21.0 ms (38%), dense 12.3 (22%, 168 GB/s), collect 6.4
+ the drafter's 0.6, drafter 6.4 (12%: its head 1.2, main_proj 0.3, 3 MTP stages), small kernels 4.3, launch gaps 2.3; experts'
sclk 2.43-2.54 GHz, the rest 2.73-2.82.

MALL (mall.hip, node1): a second plain read of 8-32 MB runs at 617-830 GB/s (cold 203-222). Nontemporal reads (the decode kernels'
weights) do not allocate (a second NT read is cold) but hit what plain loads left: 16 MB read NT in 22 us after a plain touch on
another stream vs 78 us cold (24 MB: 30 vs 115 us).

### 4. The decode MALL prefetch (k-dec2, 2026-10-20; experimental)
Why: section 3's drafts-off token leaves DRAM idle ~9.3 ms between weight-streaming launches while the next layer's dense
weights, router and shared expert are known; the served build's CUDA path forks a paced L2 prefetch at exactly those points
(round.zig prefetch), which HIP refused for want of a kernel.
- mall.hip: plain reads allocate in gfx1151's 32 MB MALL (a second read of 8-32 MB at 617-830 GB/s vs 203-222 cold); the
  decode kernels' nontemporal loads do not allocate but hit (16 MB NT read 22 vs 78 us after a plain touch).
- Harness MALL=1 (deployed kit, R = 1, single launches): glinear qa_kv 42.6 -> 23.8 us, wq_b 75.1 -> 38.6, wo_a 66.9 -> 35.1,
  wo_b 76.1 -> 40.8 after the group's weights were touched (2.7-3.3 us a MB saved).
- mallg.hip: HIP graph replay runs a forked branch serially by default (16 MB touch beside a 100 us spin: 216 us vs serial
  217, none 195); with DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1 (or ..._CLASSIC_PATH=1) beside it (139 us; with ten short main
  kernels 176-180 vs 206). The knob alone, served drafts off on b010ee4, A B B A: ~0.1 ms / token, identical hashes.
- Kernel (2927c36): tf_ds_exl3_l2_prefetch, the CUDA kernel's parameters; harness l2pf: 8 / 16 / 24 MB in 43.7 / 79.0 /
  115.7 us unpaced, 50.4 / 95.7 / 139.7 paced (engine wave_ns 4,559), NT read after 14.2 / 22.6 / 31.0 vs 39.9 / 73.7 /
  108.2 us cold, 0 bytes changed. Engine side a-arch2 4dc7ec5 (serve.conf L2_PREFETCH, default 0; sets the knob).
- Served A/B (tools q26.sh; A = current fd65f28edb87, B = kdec-pfx 6fdee0a with L2_PREFETCH=1, C = kdec-pfx with
  L2_PREFETCH=0; code / prose tok/s, 400 tokens; drafts on with FROZEN_TIMING=1):

| run | drafts off | drafts on |
|---|---|---|
| A1 | 32.60 / 32.20 | 72.17 / 41.06 |
| B1 | 33.80 / 33.54 | 73.84 / 42.29 |
| B2 | 33.70 / 33.59 | 74.02 / 42.46 |
| A2 | 32.12 / 32.34 | 72.50 / 41.26 |
| C1 | 32.25 / 32.37 | 72.54 / 41.37 |

  Drafts off B vs A +4.1% (30.94 -> 29.71 ms a token, -1.2 ms), B vs C +4.2%; drafts on +2.2% code, +3.0% prose. Token
  hashes identical in all ten runs (48951eea1fc4, 62a2c277a61d; drafts on accepted 310/401, 199/346 in every run). PPT
  median 100 W in every run; sclk median 2.79-2.89 GHz drafts off, 2.43-2.60 drafts on; load 2.1-5.2 (other workers).
  The gain is the low end of the 1.5-4 ms estimate: the prefetches overlap the gather waits and small kernels only partly
  (the ten-kernel mallg case), and the engine's budgets cap the after-wo_b window at 8 MB (the shared expert is 11 MB).

### 5. Follow-ups (not built)
- Read piece size (p-pre2's patbench, node1 ~/tfp/ppre/bench/patbench.hip: 256-byte pieces from many rows at once reach
  ~60-68% of DRAM bandwidth, contiguous 1 KB+ pieces ~98%). A decode kernel's wave reads, each k step, its 128-column
  block's 8 tiles: one contiguous piece of 8 x 4 x K2 words: 512 B (K2 4: routed experts of layers 18-22), 768 B (6: the
  other routed experts), 1 KB (8), 1.25 KB (10: most dense, shared experts), 1.5 KB (12), 2 KB (16). Dense glinear reads
  the strips layout, so a wave's k range is contiguous across steps; the experts keep the checkpoint's tile order
  ([k/16][n/16] tiles), so a wave's next step is a whole k row of tiles further on (6.9 KB for gate / up, 30 KB for down at
  K2 6). Checked with xpat.hip (2026-10-20, node1, load 0.6; the experts' read pattern alone at layer 39 / 20 shapes, 7
  places, cold copies, HIP-graph replay of 16 launches; one step of loads in flight beyond the one in use, as grouped_cp):
| K2 | matrix | piece 1 col block | 2 | 3 | 4 | flat stream |
|---|---|---|---|---|---|---|
| 6 | gate / up (31 MB) | 768 B: 222.9 GB/s | - | 2.3 KB: 222.4 | - | 240.4 |
| 6 | down (15.5 MB) | 768 B: 197.2 | 1.5 KB: 199.0 | - | 3 KB: 208.9 | 236.7 |
| 4 | gate / up (20.6 MB) | 512 B: 221.3 | - | 1.5 KB: 220.1 | - | 239.0 |
| 4 | down (10.3 MB) | 512 B: 218.4 | 1 KB: 211.1 | - | 2 KB: 207.3 | 232.3 |
  The pattern already reaches 92-94% of a flat stream for gate / up at 512-768 B pieces, and wider pieces add nothing; only
  K2 6 down gains (+6% at 3 KB), on about a third of the experts' time: at most ~0.2 ms a drafts-off token (0.6%). The
  kernel's own shortfall is elsewhere (served gate / up 203 GB/s vs the pattern's 223: decode, staging, the ready hand-off;
  down 183 vs 197). Below the 10% bar the lead set: not pursued.
- MALL prefetch budget after wo_b: 8 MB today (prefetch_budget / 2); the shared expert alone is 8.8-11 MB a layer and
  the window is ~70 us (~15 MB at 210 GB/s). Estimate +0.2-0.4 ms a drafts-off token with the prefetch on.
- The >8-row glinear regression (row-invariant dot2 rows path 22-31% slower than WMMA at 16-63 rows): ~7-8 ms of TTFT on
  17-63-token prompts; nothing for single-stream decode. Fix keeping invariance: WMMA 16-row passes for prompt-chunk
  windows only (an args flag; decode / verify windows keep the 1-row order).

### 6. The dense gap and the best remaining decode lever (k-dec2, 2026-10-20; from section 3's traces, nothing run)
Dense glinear, drafts off, rank 1, per token: 11.31 ms for 2,063 MB = 182 GB/s; at a flat stream's ~235 GB/s (xpat.hip)
8.78 ms, so 2.53 ms over. By launch group (time; MB a launch x launches; GB/s; over 235 GB/s, and over a launch):
| group | ms | MB x n | GB/s | over (ms) | over a launch (us) |
|---|---|---|---|---|---|
| wo_a (4 groups) | 2.41 | 10.5 x 40 | 174 | 0.63 | 15.6 |
| wq_b | 2.82 | 13.1 x 40 | 186 | 0.59 | 14.8 |
| wo_b | 2.72 | 13.1 x 40 | 193 | 0.49 | 12.2 |
| wq_a + wkv | 1.29 | 5.8 x 36 | 161 | 0.41 | 11.3 |
| head | 1.23 | 248 x 1 | 202 | 0.17 | 172 |
| Engram wkv | 0.49 | 44 x 2 | 181 | 0.11 | 56 |
| compressor / indexer groups | 0.35 | 0.1-9 x 13 | 5-181 | 0.14 | 6-15 |
The gap is a fixed ~11-16 us a launch (ramp, split-K epilogue, tail) on ~165 launches of 5-13 MB, i.e. the small-matrix
structure of TP2 decode, not a bandwidth shortfall inside the launches; the launch gaps between kernels (2.2 ms) are on top
and not counted here. Part of it is already recovered when the MALL prefetch is on (qa_kv, wq_b, wo_a read from the MALL:
section 4's -1.2 ms), so with the prefetch on the dense gap left is ~1.3 ms at most.
Bit-identical levers left and their estimates (per drafts-off token, ~30 ms):
- Dense launch fusion keeping every sum's order: wo_a + wo_b as one launch (wo_b's blocks wait on wo_a's ready count, as
  grouped_cp's down waits on gate / up, loading their first k steps meanwhile), rot_many folded into the qa group's glinear
  (each block rotates its own k range), the q norm folded into wq_b's prologue: ~8-10 us a fused boundary plus the
  removed launch gaps, ~0.6-0.9 ms (2-3%), less with the prefetch on. Multi-hour, with the forward-progress risk of
  spinning blocks.
- Expert kernel internals: served gate / up 203 GB/s against the read pattern's 223, down 183 against 197: ~0.8 ms if
  closed entirely; k-dec's seven experiments (section 2) recovered none of it; realistic ~0.3-0.4 ms (1%).
- Head plan (SK 2) and the per-shape split table change the sums (not bit-identical); 77df5c7 showed no served gain.
Best remaining decode lever in the kernels: dense fusion, ~0.6-0.9 ms (2-3%), under the lead's ~3% bar: not started.
Larger decode items are outside these kernels: gathers 3.2 ms (a-arch2's fold, transport), launch gaps 2.2 ms and small
kernels 4.1 ms (aux kernel fusion).

### 7. Small-kernel fusion: scope (k-dec2, 2026-10-20; from section 3's drafts-off trace, rank 1, 765 rounds; nothing run)
Per drafts-off token: 1,112 launches; small kernels 4.10 ms; launch gaps below 20 us 2.21 ms (~1.95 us a launch).
| group (per site / layer) | launches | ms | mean us | fusable keeping the bits |
|---|---|---|---|---|
| hc mix chain, 80 sites (attn + ffn mix): _hc_post_only > _hc_finish_s > _hc_mix_part > _hc_sinkhorn | 317 | 2.08 (0.14 + 0.33 + 1.29 + 0.32) | 1.9 / 4.1 / 16.1 / 4.0 | yes, as one kernel a site if finish / sinkhorn keep their op order (the parts summed in part order by the last block, as glinear's split-K); _hc_post_only is already in a-arch2's COLL_FOLD (tf_ds_coll_post) |
| router top-k _route_parts (+ decode_prep 0.17 ms, mine) | 40 + 40 | 0.70 + 0.17 | 17.5 / 4.1 | yes: _route_parts into decode_prep (exact selection; the weights' normalization in its order) |
| sparse attention + _sparse_attn_merge_rot2 | 40 + 40 | 0.52 + 0.13 | 12.9 / 3.3 | yes: the merge by the attention's last block, parts in order |
| _q_kv_norm (q norm, kv norm + rope, wq_b's rotated rows) | 40 | 0.15 | 3.7 | partly: the q half into wq_b's prologue (each block re-norms the 1,280-wide row in the kernel's order); the kv half stays |
| exl3 rot_many (qa / head / Engram inputs) | 47 | 0.11 | 2.4 | yes: each glinear block rotates its own k range (rot128 is per 128-block) |
| indexer (5 layers: rope_heads, fp4qd, rowmm2, index_score, topk_sel), compressor, Engram, misc | ~75 | ~0.42 | 2-34 | not worth it (few launches) |
| gather stage (w-link) | 83 | 0.40 | 4.8 | a-arch2's transport / fold work |
Gaps: each removed launch takes its ~1.95 us gap with it; the rest (~1.95 us x the launches left) is graph replay's
per-node cost regardless of fusion. The set below removes ~360 launches (~0.7 ms of gaps); ~750 launches (~1.5 ms) stay.
Best bit-identical set and estimate (per drafts-off token ~30 ms): hc chain 3 -> 1 a site (after COLL_FOLD takes
_hc_post_only): 160 launches, ~0.3 ms gaps + ~0.35 ms of the tiny kernels' ramp / tail; _route_parts into decode_prep:
~0.15 ms; attention merge into the attention kernel: ~0.15 ms; rot_many and the q norm into glinear: ~0.35 ms. Total
~1.3-1.5 ms drafts off (4-5%); drafts on ~1.6-1.9 ms a 55 ms round (~3-3.5%: the drafter's 3 MTP stages have the same
chains, the rest of a round is unchanged). Build ~10-14 h across aux_hc.hip / aux_attn.hip (w-tri's ports), exl3.hip
(mine) and the call sites (tri_hc.zig, round.zig), plus ~2 h of gates (each fused kernel bit for bit against the chain it
replaces on real tensors, the layer gate, the four Spark logits SHAs, served A/B). Main risk: reproducing the Triton
kernels' exact reduction order in the fused kernels (any slip changes output; the logits SHAs catch it), then the serial
tail a last-block reduction adds, which can eat a tiny kernel's saving.
### 8. The hc mix chain fused (k-dec2, 2026-10-20; branch kdec-hc, TF_HC_FUSE=1, default off)
_hc_fused (kdec/hc/hc_fuse.py) replaces hc_dots' _hc_finish_s (role 1) + _hc_mix_part + _hc_sinkhorn with one launch, from
tri_kernels.py's own code at 8 warps; the 2-warp mixes' sum of squares is reduced in that kernel's tree spelled out
(_ss_tree), else part slot 24 differed. Kit: gfx1151-v5 + 6 _hc_fused variants (kdec/hc/aot_fused.py).
- Bits: bench 36 / 36 cases identical (rows 1-6, three sites' real weights, synthetic streams); the two-rank layer gate
  (gpufix2-p24-rounds, strict) dumps 1,962 points on each rank, identical between TF_HC_FUSE 0 and 1; the four Spark
  prompts' logits SHAs 893b029275e4 / 91c39a0995c2 / 80daaffffbe9 / 5eda4bab4013 with 0 and 1; served code / prose
  48951eea1fc4 / 62a2c277a61d in all eight A/B runs, drafts off and on. A traced run shows 77 _hc_fused a round.
- Served A/B (deploy-cand/8526b22db6ad, A = 0, B = 1, code / prose tok/s; drafts on at the production timing):
| run | drafts off | drafts on |
|---|---|---|
| A1 | 32.42 / 32.52 | 72.22 / 41.19 |
| B1 | 33.24 / 33.38 | 72.15 / 41.42 |
| B2 | 33.20 / 33.23 | 72.11 / 41.20 |
| A2 | 32.44 / 32.63 | 72.62 / 41.46 |
  Drafts off +2.3% (30.77 -> 30.06 ms a token, -0.7 ms); drafts on -0.4% code, 0.0% prose. PPT 100-103 W, sclk 2.82-2.88
  GHz drafts off, 2.48-2.60 drafts on, load 2.2-3.4.
- Why drafts on gains nothing (rank 1 traces, a site's time): at 1 row the fused launch takes 18.3 us served (bench 14.6)
  against the chain's 4.1 + 16.1 + 4.0 us + two more launch gaps; at 3-6 rows it takes 23.5-27.7 us against the chain's
  ~26-28.5 with gaps: a row group's finish rows run one after the other in one program, and its Sinkhorn rows one after the
  other in the last mix program to arrive, where the chain runs both a program a row. Fix (not built): finish programs a
  row (grid y NB + RB), and at 2+ rows the Sinkhorn back in its own launch (3 -> 2 launches); estimate ~4-6 us a site at
  verify widths, ~0.3-0.5 ms a 55 ms round (<1% drafts on).
- The lead's extra checks: (1) the arrival counters over long sessions (kdec/hc/kd_session.sh + hc_session.py, TF_HC_FUSE=1,
  one server each): drafts off 15 requests / 4,842 one-row rounds and drafts on 15 requests / 1,924 verify rounds (widths
  2-6), each with an early stop (max_tokens 7), a stream abandoned after 40 events (cancelled between rounds) and a stop
  sequence; code / prose 48951eea1fc4 / 62a2c277a61d before and after, in both (a stale count would leave a Sinkhorn
  unrun or early and change them). (2) Real activations with TF_HC_FUSE=1: the layer gate's dumps and the four logits
  SHAs above. (3) Kit identity: kit-gfx1151-v5hc's 199 non-_hc_fused aot.json entries equal gfx1151-v5's field for field,
  their 199 objects byte-identical; in the deployed dsv41 kits (current 6bdf46c78848 vs the candidate) every native
  object's code is identical (disassembly), the files differing only in .strtab (HIP's per-path __hip_cuid symbol).
- The counters read directly (kdec/hc/cnt_test.py, node1): one HIP graph of 120 _hc_fused launches at rows 1-6 in turn over
  20 sites' real weights, replayed 10 times (1,200 consecutive launches): all 64 counters 0 afterwards; each width's last
  launch bit-identical to the 3-kernel chain on the same inputs.

### 9. Decode kernels at 12-16 rows (multi-agent rounds; k-dec2, 2026-10-20; from code and existing data, nothing run)
Context (a-arch2's telemetry): rounds at C2 carry ~12 rows, at C4 / C8 the 16-row cap; round time +~4.75 ms a row past 6
(61.5 ms at 6 rows, 90.5 at 12, 109 at 16). Sources: section 3's drafts-on trace split by rows (rank 1, rows 2-6, 285 rounds;
traced, ~5% above untraced), k-dec's harness scans (glinear w_gl_* R 8-63, experts w_x39_* R 8-63, uniform routes).

Per row of a round, rows 2 -> 6 (drafts-on trace): experts +4.21 ms, gathers (stage + collect) +0.67, dense glinear +0.31,
attention +0.14, hc chain +0.10, router +0.085, the head 0, the drafter 0 (traced: +5.5 ms a row in all).

| group | code path at 12-16 rows | row-invariant | ms a round at 6 rows (trace) | est. at 16 rows (C4 = 4 x 4) |
|---|---|---|---|---|
| routed + shared experts | exl3.experts.grouped_cp: a place (distinct expert) per program set, every member row in the 1-row order (dot2 rows path; the shared expert has all 16 rows: the two-part 9-16-row path) | yes | 31.7 | ~75 (~59 at 12 rows): ~59 places a layer x ~31 us |
| dense glinear (qa, wq_b, wo_a, wo_b, idx) | exl3.glinear below EXL3_WIDE_MIN (64): dot2 rows path, two parts of 4 tiles at 9-16 rows (the ">8-row regression") | yes | 11.8 | ~17.9 (harness 16 / 6 rows: 467 / 307 us) |
| the head | exl3.glinear, the same two-part path | yes | 1.22 | ~1.6 (1,569 / 1,179 us) |
| attention (sparse_attn, merge, q/kv norm, rope) | aux.sparse_attn (HIP) + Triton merges, any rows | n/a | 1.43 | ~2.8 |
| hc chain (post_only, finish_s, mix_part, sinkhorn) | Triton, _hc_mix_part RB 4 (9-12 rows) / 8 (13-16) | n/a | 2.19 | ~3.2; TF_HC_FUSE=1 is slower here (a row group's finish and Sinkhorn rows run one after the other: 8 at 13-16 rows) |
| router (_rowmm_parts, _route_parts) | Triton | n/a | 1.84 | ~2.7 |
| gathers | tcp_proxy stage / collect | n/a | 8.6 | ~13-15 (a-arch2) |
| drafter | DSpark block per drafted stream | - | 5.5 (one stream) | not modelled (per stream or batched: a-arch2) |

Why ~4.75 ms a row: the experts. Each added row adds up to 6 routed slots; the round reads every distinct expert once
(a place), so the cost follows distinct experts a layer, D. Harness (L39, 3-bit): t_layer ~ 0.26 ms + 31 us x (places - 7)
(2-bit layers 18-22: 29 us), at ~213 GB/s (93% of the 230 GB/s read ceiling) up to 83 places; dot2 is 3% faster than the
old WMMA there (w_x39_base). Served routing is correlated within a stream: 6 rows read ~24 places a layer (31.7 ms / 40 /
31 us), against ~35 for uniform routes; 4 rows ~16. Streams route independently, so a round of S streams of r rows reads
~384 (1 - (1 - d_r / 384)^S) + 1 places (d_r one stream's distinct routed experts at r rows): C2 x 6 rows ~46, C4 x 4
rows ~59, uniform-random 16 rows ~83. That is ~3.5-4 ms a row past 6 of the ~4.75; dense (two-part path) ~0.4-0.6,
gathers ~0.6, the small kernels ~0.3. At 16 rows a rank reads ~15 GB of expert weights a round.
What to look for in a-arch2's distinct-experts counter: D a layer against rows and streams. If D tracks the union model
(~23 / 45 / 58 routed at 1 x 6 / 2 x 6 / 4 x 4) and the experts' time a layer tracks 0.26 + 0.031 (D + 1 - 7) ms, the
multi-agent rounds are bound by reading distinct experts at ~93% of the read ceiling: no expert-kernel change recovers
much; the levers are fewer distinct experts a round (round composition) or fewer bytes an expert.

Batch-invariant fixed-order WMMA family (one arithmetic at every width, 1 row padded; dense glinear and the head only;
the experts keep the dot2 rows path, which is faster at every width and already row-invariant):
- Dense groups at 16 rows: WMMA 372 vs dot2 467 us (-20%); head 1,199 vs 1,569 (-24%) (w_gl_base vs w_gl_new5). At 2-6
  rows the old WMMA passes were 0-2.5% slower than dot2 (section 1); 1-row WMMA was never timed (the wmall object is
  built, kdec/mkexp2.py, not run; it compiles to 192 VGPRs / 8 waves a SIMD against 231 / 6).
- Estimate: ~-4 ms a 16-row round (dense -3.6, head -0.4: ~3.5-4% at C4); ~-2.5 ms at 12 rows; 1 row: +0 to +0.3 ms a
  token (<= 1%) until measured.
- Work: kernel only (glinear's decode passes all on the WMMA strip, as wmall); no engine change. Gates: rowinv_gl
  (synthetic, real), contracts at 1-63 rows, the layer gate, new reference logits SHAs, served A/B at C1-C6.
- IT CHANGES OUTPUT (new bits at every width, drafts on still = drafts off): needs sign-off and the quality set rerun.

Exact-output levers at 12-16 rows (today's bits): the MALL prefetch (L2_PREFETCH=1, built, experimental): its dense groups
read from the MALL, est. ~1-1.5 ms a round; nothing in the expert kernel (bandwidth-bound on distinct experts); a
single-part 16-row glinear path would spill (k-dec's first row-invariant version did), not estimated. Keep TF_HC_FUSE off
for multi-agent (its verify-width fix, finish a row and the Sinkhorn in its own launch, is worth ~0.3-0.5 ms a round).
