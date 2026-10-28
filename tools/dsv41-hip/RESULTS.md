# RESULTS: DeepSeek-V4.1-Flash-1017 on two Strix Halo boxes (TensorFold port)

Handoff §10 deliverable: measured speed, correctness coverage, resource use, reference provenance and remaining risks,
each with the file that holds the raw data. As of 2026-10-19 23:45 UTC. Speed is measured on strix-port **86302cf**; the correctness rows below name their deployment (mostly **58fd1a9**) on both
boxes (kit v4, the router fix, exl3.gemm and the prompt experts), configuration `q-58s.conf` = the deployed serve.conf
with GPU_STRICT=1 and a 16,384-token pool (TP2 over USB4 tcp_host on both links, MTU 1500, 1 stream, greedy, drafts
off); `q-58g.conf` adds HIP graphs over tcp_proxy. Earlier rows name their own deployment. The gate-by-gate state is `PORT_STATUS.md` (rendered
from `gates.json`); `qualify.sh` gives the verdict. Nothing here is projected unless it says so.

## 1. Speed

### Matched against the two-Spark run, in its own metric (86302cf, 2026-10-19 23:24-23:41 UTC)

Deployment strix-port **86302cf** (proxy + HIP graphs, prefill split 256: the shipped serve.conf) with CONTEXT 16384,
quiet windows on both boxes, the creator's prompt set v1 and filler method, temperature 0, thinking off, every prompt
cold, 5 repetitions, 0 errors, 0 kit fallbacks; each run's manifest.json is complete (commit, binaries, 168 kit objects,
bundle, both ranks' resolved configuration, toolchain, prompt token ids). Spark: recipe 9fa230d3 runs/2026-10-18-1017-
tensorfold-tp2/results/tp2a (blob 06dd8054). Decode = (completion tokens - 1) / (end - first content event).

| category | drafts off | drafts on (provisional: output not yet reproducible) | Spark | off / Spark | on / Spark |
|---|---|---|---|---|---|
| coding | 32.30 | 76.00 | 105.63 | 0.31 | 0.72 |
| json | 32.00 | 59.24 | 92.12 | 0.35 | 0.64 |
| narrative | 31.80 | 38.45 | 61.60 | 0.52 | 0.62 |
| prose | 31.91 | 41.03 | 55.43 | 0.58 | 0.74 |
| math | 32.15 | 65.22 | 99.55 | 0.32 | 0.66 |
| reasoning | 31.84 | 54.96 | 82.00 | 0.39 | 0.67 |
| summary | 31.68 | 40.93 | 62.87 | 0.50 | 0.65 |
| format | 32.08 | 76.59 | 114.09 | 0.28 | 0.67 |
| **8-category mean** | **31.97** | **56.55** | **84.16** | **0.38** | **0.67** |

| prompt reading (one output token, whole request) | drafts off | drafts on | Spark | off / Spark | on / Spark |
|---|---|---|---|---|---|
| 2,952 tokens (creator target 2000) | 1,085.7 | 1,146.8 | 1,921.6 (2,950) | 0.56 | 0.60 |
| 11,594 tokens (creator target 8000) | 1,251.7 | 1,297.0 | 2,180.2 (11,592) | 0.57 | 0.59 |

Medians; ranges, whole-request tok/s, TTFT, tokens per round and acceptance are in node1 `~/tfp/qual/out/baseline-86302cf.md`
and the runs' bench-*.json (`~/tfp/qual/out/bench-1019T232410-q-86-off`, `bench-1019T233249-q-86-on`).

### Earlier measurements

| Measurement | Result | Conditions | Raw data |
|---|---|---|---|
| Streamed decode, code / prose, 400 tokens | 25.14 / 25.32 tok/s (lead's client) | aa0e5cf, drafts off, quiet | node1 `~/tfp/run/lead-lead4/stream.txt` |
| First light (ad3b8c5 natives) | 20.62 / 20.35 tok/s | 2026-10-19 10:16-10:25 PDT | `handoff-2026-10-19/01-LAST-STEPS.md` |
| 8,165-token prompt at STRICT=1 | prefill 829 tok/s, decode 24.0 tok/s | 58fd1a9, the lead's run | node1 `~/tfp/run/lead_fl10.log` |
| HIP graphs over tcp_proxy vs eager | a 29-token reply 1.245 s vs 1.41 s, token ids identical | 58fd1a9, not quiet | `~/tfp/qual/out/qs-1019T213921/` |

Reference (PLAN.md): 2 x DGX Spark, same EXL3 2.9 bpw pack, TP2. the target bands: prose decode at 65-75% of Spark and prompt
reading at 70-85%. Prose with drafts on is at 74%; narrative (62%) and prompt reading (59-60%) are below their bands.

## 2. Correctness coverage (summary; details and evidence paths in PORT_STATUS.md)

| Area | Result | Evidence |
|---|---|---|
| Inputs | checkpoint (66 files) and Engram shards authenticated; both rank caches byte-checked in full; chat template, tokenizer, token map, Engram constants, RoPE equal their pinned sources | M0 gates |
| Prompt and reply rendering | 580 named cases + 5,000 fuzz conversations token-identical to the official encoder through the real server; 1,398 replies parsed as the official parser does | P2.prompt-parity, P2.stream-parity |
| GPU programs | smoke 14/14, kernel ABI 5/5, glue 11/11 from the deployed binaries on **both** boxes (aa0e5cf) | `~/tfp/qual/out/gpu-aa0e5cf/` |
| EXL3 decode | exact for all 65,536 states and every K on real tensors (prototype); the engine's unpack bit-exact vs the oracle; full-pack oracle cross-check 44,500/47,900 identical so far | M1.exl3-exact (receipt from the deployed kit still owed) |
| Auxiliary kernels | 844 fp64-reference checks pass; the router's softplus fixed (k-tri 4d59b93: fp64 grid over lg in [-100, 30], 44/44 router contract cases) | M2.aux-operators pass |
| Layer trace vs FP64 reference (P=160, the minimum trace) | on 58fd1a9 at STRICT=1: routing 0 violations in 12,240 rows, indexer 0 of 2,384 rows differ, logits nrmse 2.85e-2 (threshold 8.35e-2), every attention/MoE/Engram/head point in bounds, the 6-row verify window graded; **fail**: layer 39's output stream has a systematic -1.2% scale (both ranks) | M3.trace-target **fail**, `~/tfp/qual/out/trace-p160-58fd/` |
| Sampling contract | 6,000 rows token-identical to the pinned contract (exact_sampling.py @ 222a9c0b); 12/12 distributions pass chi-square; 3 negative controls fail as they should | `~/tfp/qual/out/sampling-1/` |
| Served functional suite | 58fd1a9, STRICT=1: 13 pass / 4 fail / 1 skipped: every prompt size 5-20 and 127-8,000 rows, tools (auto), thinking, stop, seeded sampling, concurrent, cancel pass (U42 closed); exact capacity -> 500 and max_tokens -3 accepted (U43); structured output and required tools not in this deploy (U44/U45, merged 6c5962a) | M3.served-functional **fail** |
| Served faults | kill either rank mid-decode: the client gets an error (PeerLost in 49 ms / closed in 0.3 s), no token after the fault; 5 peer mismatches (round_rows, aux, streams, pool, kit) refused before serving, naming the field; a mismatched bundle refused by the launcher. **Fail**: a stalled rank never exits after it resumes, /health stays "ok" after a partner failure, rank 0 waits forever for a peer that died during load (U47); a rank cache under another content key is accepted (U16) | `~/tfp/qual/out/qf-1019T*/faults-regraded.json` |
| HIP graphs | served over tcp_proxy at STRICT=1: 32 replies identical to eager; capture/replay counters not yet exposed | M4.graphs unrun |
| No NVIDIA code | the serving processes map only ROCm and system libraries; no NVIDIA library installed | M1.no-nvidia-runtime |
| Clean restart | stop both ranks, start again: identical token ids for the reference requests | M5.clean-restart |

## 3. Resource use (served, aa0e5cf, 1 stream, 8K context)

| Quantity | node0 (rank 0) | node1 (rank 1) | Source |
|---|---|---|---|
| GPU-mapped memory (GTT) after load | 98.79 GiB, unchanged while serving (8K and 16K pools) | 98.79 GiB, unchanged | memledger, q1 and qs-1019T211208 runs |
| MemAvailable, lowest while serving, idle box (58fd1a9, 16K pool) | 20.45 GiB | 18.36 GiB | qs-1019T211208 |
| Rank process RSS | 1.0 GiB serving, 5.0 GiB peak while loading | 0.8 / 4.2 GiB | same |
| MemAvailable, lowest while serving | 9.5 GiB (w-ref's capped CPU jobs resident) | 20.8 GiB | same |
| Swap-out, OOM, memory.max events | none | none | same |
| Arena (rank 1) | 2.54 GiB used / 2.75 GiB reserved | | rank 1 log |
| Package power / GPU temperature / CPU Tctl (short suite) | <= 46.8 W / <= 48 C / <= 56.5 C | <= 49 W / <= 48 C / <= 61.8 C | same |

Rank 0 needs an otherwise idle node0: in one fault-matrix start it refused to load ("98.9 GiB of weights do not fit
the 97.5 GiB the HIP memory budget grants") while node0's background jobs held memory (U25).

## 4. Reference provenance

- FP64 model reference: DeepSeek's own `inference/model.py` at the pinned 1017 revision 1bae0e10 run in FP64, EXL3
  decoding from ExLlamaV3 v1.6.2 (MIT); no TensorFold code (w-ref, node0 `~/tfp/ref`). Contracts: numerics.json, copied
  with its history and controls into `tools/dsv41-hip/numerics/` (PROVENANCE.md). Thresholds were frozen from FP64-vs-twin
  calibration (1.2.0) before any GPU result was graded; the P=24 grade used 1.6.0, the P=160 trace 1.7.0, a
  recalibration on fixture set v2 that raised 61 thresholds and is open for the reviewer's sign-off.
- Sampling contract: `src/tensorfold/engine/exact_sampling.py` at the pinned CUDA source 222a9c0b, plus distributions
  derived from the definitions alone.
- Prompt-rendering reference: the checkpoint's `encoding/encoding.py` at 1bae0e10 and its tests; the creator's chat
  template c282975e.
- Benchmark prompts and client conventions: recipe 9fa230d3 `bench/prompts-v1.json` (git blob 15f511f8) and
  `runs/2026-10-17-1017-tensorfold/tools/v41bench_tf.py` (blob d7d0d227), fetched from the pinned commit and checked
  against its tree.

## 5. Remaining risks

1. The layer-39 output scale in the FP64 trace (-1.2%, both ranks): a contract failure until localized and fixed; the
   logits pass only because the final norm removes a uniform scale.
2. Speed: decode is about 26 tok/s and 8K prefill 829 tok/s without drafts; the targets are 35 tok/s decode with
   drafts and 1,000 tok/s prefill at 8K. DSpark is not yet served, so M4 (draft/ordinary equivalence, commit/rollback,
   the AMD cost profile) is unrun.
3. Exact-capacity requests are refused with a 500 and invalid max_tokens is accepted (U43).
4. Failure handling: stalled ranks linger and startup waits forever for a dead peer (fault matrix); rank 0 keeps
   answering 500 after its partner fails.
5. Handshake identity is narrower than the handoff's (no engine commit, bundle, KV precision, GPU_STRICT) (U46).
6. Rank-0 memory margin on node0 depends on the box being idle (U25).
