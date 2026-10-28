# numerics/: the numerical contracts the port is graded against (handoff section 7)

Copied here by q-qual on 2026-10-19 from w-ref's reference repository (node0 `~/tfp/ref`, HEAD 1067cf0), so that the
contract every grade cites is in strix-port. w-ref's repository stays the source; these files are byte copies.

| File | Version | SHA-256 | Origin |
|---|---|---|---|
| `numerics.json` | **1.7.0** | fd1c153d4256103acfb2b8a3172254af803d959ae52423359d7e3dda90047365 | working copy written 2026-10-19 14:00:33 PDT by `tools/run_final_cal.sh` (044dde91...) / `numerics/make_numerics.py` (d1fc5167...); not committed in w-ref's repository |
| `layer_points.json` | 1.7.0 | c89080523127e4c4f4f6094c6de99a31a5af054fe3351dd786d7d8d3510c89be | same run |
| `op_points.json` | 1.7.0 (new) | 33109397cd0667e882ac10a4403898bdc6f613fae46d6619fd6fe8b2bd08ec9d | same run |
| `controls.json` | 1.7.0 | b16c82d017b1e448eef59e59475dda73be7490a861aa798bf4a304080433cea4 | same run |
| `numerics-1.6.0.json` | 1.6.0 | 87f829df44e93c668d331ed4467296e925658fb89aaf8cac3b609646df2e719c | w-ref commit 8c14931 (2026-10-19 03:54 PDT) |
| `layer_points-1.6.0.json` | 1.6.0 | 4599cc2a1ac4e5a3acad2e7f6d015ec8e9bd99355e2325e677f447fdfed36e76 | w-ref HEAD |

## History (from the changelog and w-ref's git log)

1.0.0 (00:29 PDT) per-operator contracts, written before any optimized kernel was evaluated; 1.1-1.1.1 ambiguity
bands and budget simulations on real inputs; **1.2.0 (01:53) layer and trace thresholds frozen from calibration C1
(FP64 vs native-precision twins) by a fixed rule (3 x the worst twin value) before any GPU layer or trace result was
examined**; 1.3-1.6 router grading against the pack's F16 gate, NaN contract, GPU fixture schema, layer-gate controls,
the gfx1151 WMMA rule. The first GPU layer grade (P=24, first light) used 1.6.0.

**1.7.0** (14:00) changes budgets, so it is a review item (handoff section 7: a budget change invalidates prior passes
and needs review): (1) grading against the served prompt path's bounded replay (integration-owner decision); (2)
layer_points recalibrated by the same 1.2.0 rule on fixture set v2 (decode rounds in the RoundDecoder.run layout, bf16
taps, gathers graded on their token rows): **61 thresholds rose and 36 fell**; the largest rises are row-level:
`bf16/moe_gather/row_nrmse` 0.139 -> 1.038, `bf16/moe_partial/row_nrmse` 0.394 -> 1.273, `bf16/h_out/row_nrmse` 0.313 ->
0.881 (the fp32 tap thresholds, previously 0, now ~0.005); (3) new operator-replay thresholds (op_points); (4) the
reference evaluates indexer scores in row blocks above 2^27 elements (memory only).

## Controls under 1.7.0 (controls.json)

| Control | Expected | Result |
|---|---|---|
| layer gate, bf16 twin / fp32 twin | PASS | PASS / PASS |
| layer gate, ranks swapped in attention | FAIL | FAIL |
| layer gate, attention sign flipped at layer 7 | FAIL | FAIL |
| layer gate, h_out scaled 1% | FAIL | FAIL |
| layer gate, MoE rank 1 zeroed / MoE scaled 1% at layer 7 | FAIL | FAIL / FAIL |
| layer gate, attention scaled 1% at layer 7 | FAIL (sensitivity) | **PASS: below the layer gate's resolution** |
| operator replay, MoE rank 1 zeroed / MoE scaled 1% | FAIL | FAIL / FAIL |
| operator replay, MoE scaled 0.1% | (resolution probe) | PASS |

Open review items for the independent reviewer: the threshold rises above (whether the v2 twins justify them), and the
attention 1% scale control, which no gate detects (no operator-replay control for attention was run).

## Grades that used these files

- P=160 trace on 58fd1a9 (`node1:~/tfp/qual/out/trace-p160-58fd`): numerics.json fd1c153d (1.7.0), grade.py ac7f0759,
  recorded in its `grader.sha256`.
- P=24 layer gate on 009fe20 / 05832c5 (first light): 1.6.0.
