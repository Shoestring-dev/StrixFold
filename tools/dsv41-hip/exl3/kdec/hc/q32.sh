#!/bin/bash
# q32 (k-dec2, hc fusion stage 3) on deploy-cand/8526b22db6ad: gates with TF_HC_FUSE 0 and 1 (the two-rank layer gate,
# its dumps compared point by point; the four Spark prompts' logits SHAs drafts off; code / prose hashes), then the
# alternating served A/B (drafts off A B B A, drafts on A B B A; A = 0, B = 1). Plain user unit (the ranks start here).
cd ~/tfp/kdec
D=deploy-cand/8526b22db6ad
log() { echo "$(date -Is) q32 $*"; }
for f in 0 1; do
  log "layer gate fuse $f"; hc/kd_gate.sh hcg-f$f $D $f > logs/hcg-f$f.out 2>&1; log "layer gate fuse $f rc=$? $(tail -1 logs/hcg-f$f.out)"
done
for b in node0 node1; do
  r=$([ $b = node0 ] && echo 0 || echo 1)
  if [ $b = node0 ]; then ssh -n node0 "python3 - ~/tfp/kdec/gate/hcg-f0/dump/rank0/layers.jsonl ~/tfp/kdec/gate/hcg-f1/dump/rank0/layers.jsonl" < hc/cmp_dump.py; else python3 hc/cmp_dump.py gate/hcg-f0/dump/rank1/layers.jsonl gate/hcg-f1/dump/rank1/layers.jsonl; fi | sed "s/^/q32 dump rank $r: /"
done
Q="'"
for f in 0 1; do
  G=ppgate/conf/hc-l$f.conf
  cp ~/tfp/$D/tools/dsv41-hip/serve.conf $G
  sed -i -e "s|^DEPLOY_DIR=.*|DEPLOY_DIR=${Q}\$HOME/tfp/$D${Q}   # k-dec2 gate|" -e "s|^CONTEXT=[0-9]*|CONTEXT=16384   # k-dec2 gate|" \
    -e "s|^LOGITS_SHA=.*|LOGITS_SHA=1   # k-dec2 gate|" -e "s|^BENCH=1|BENCH=0|" -e "s|^MAX_TOKENS=[0-9]*|MAX_TOKENS=512|" \
    -e "s|^DRAFTS=.*|DRAFTS=0   # k-dec2 gate|" -e "s|^L2_PREFETCH=.*|L2_PREFETCH=0   # k-dec2 gate|" \
    -e "s|^RUN_ROOT=.*|RUN_ROOT=${Q}\$HOME/tfp/kdec/ppgate/runs${Q}|" $G
  if [ $f = 1 ]; then sed -i -e "s|^WRAP0=.*|WRAP0=${Q}/usr/bin/env TF_HC_FUSE=1${Q}|" -e "s|^WRAP1=.*|WRAP1=${Q}/usr/bin/env TF_HC_FUSE=1${Q}|" $G; fi
  scp -q $G node0:tfp/kdec/ppgate/conf/hc-l$f.conf
  bash ppgate/bin/kd_serve.sh hc-l$f hc-pp-l$f --targets 2000,8000 --reps 2 > logs/hc-pp-l$f.out 2>&1
  log "logits fuse $f: $(grep -o '"sha256": "[0-9a-f]\{12\}' ppgate/out/hc-pp-l$f/rank0.log | cut -c12- | tr '\n' ' ')"
done
W="WRAP0='/usr/bin/env TF_HC_FUSE=1'"
W1="WRAP1='/usr/bin/env TF_HC_FUSE=1'"
for dr in 0 1; do
  for r in A1 B1 B2 A2; do
    n=hc-d$dr-$r
    if [ ${r:0:1} = B ]; then tools/budget_run.sh $n $D 0 BENCH_CASES=code,prose DRAFTS=$dr "$W" "$W1" > logs/$n.out 2>&1
    else tools/budget_run.sh $n $D 0 BENCH_CASES=code,prose DRAFTS=$dr > logs/$n.out 2>&1; fi
    log "$n: $(grep -o 'tokens=[0-9]* sha=[0-9a-f]*' runs/$n.log | tr '\n' ' ') $(grep -o 'decode_tok_per_s": [0-9.]*' runs/$n.log | head -2 | tr '\n' ' ') $(grep -o 'uptime": "[0-9. ]*' runs/$n.log | head -1)"
  done
done
log done
