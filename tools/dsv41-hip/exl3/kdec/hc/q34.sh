#!/bin/bash
# q34 (k-dec2, after q32): proof the fused path runs served: one drafts-on run of deploy-cand/8526b22db6ad with
# TF_HC_FUSE=1 and rank 1 under rocprofv3 (kernel trace): the round's kernel counts (_hc_fused vs the 3-kernel chain).
cd ~/tfp/kdec
true
D=deploy-cand/8526b22db6ad
mkdir -p prof/hc-trace-on
tools/budget_run.sh hc-trace-on $D 0 BENCH_CASES=code DRAFTS=1 "WRAP0='/usr/bin/env TF_HC_FUSE=1'" \
  "WRAP1='/usr/bin/env TF_HC_FUSE=1 /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv -d ~/tfp/kdec/prof/hc-trace-on --'" > logs/hc-trace-on.out 2>&1
F=$(find prof/hc-trace-on -name "*kernel_trace.csv" | head -1)
echo "$(date -Is) q34 trace $F: $(grep -o 'tokens=[0-9]* sha=[0-9a-f]*' runs/hc-trace-on.log | tr '\n' ' ')"
for k in _hc_fused _hc_finish_s _hc_mix_part _hc_sinkhorn _hc_post_only _embed_init; do echo "q34 $k $(grep -c "\"$k" $F)"; done
echo "$(date -Is) q34 done"
