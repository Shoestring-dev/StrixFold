#!/bin/bash
# q26 (k-dec2): served A/B for the MALL prefetch. A = deploy/current resolved at the start (its serve.conf); B = deploy-kdec/<kdec-pfx>
# with L2_PREFETCH=1; C = the same deploy with L2_PREFETCH=0 (the build without the prefetch: separates the other merges).
# Drafts off: A B B A C; drafts on (FROZEN_TIMING=1: draft widths from the tables, so the runs take the same rounds):
# A B B A C. Code + prose, 400 tokens; sclk / PPT sampled (budget_run.sh). Plain user unit (the ranks start here).
cd ~/tfp/kdec
until grep -q "^exit=" logs/dep-pfx.log 2>/dev/null; do sleep 30; done
grep -q "^exit=0" logs/dep-pfx.log || { echo "deploy failed"; exit 1; }
A=deploy/$(basename $(readlink -f ~/tfp/deploy/current))
P=deploy-kdec/$(grep -o "deployed [0-9a-f]*" logs/dep-pfx.log | tail -1 | cut -c10-21)
echo "$(date -Is) q26 A deploy $A, B/C deploy $P"
for dr in 0 1; do
  X=""; [ $dr = 1 ] && X="FROZEN_TIMING=1"
  for r in A1 B1 B2 A2 C1; do
    case ${r:0:1} in A) D=$A; E="";; B) D=$P; E="L2_PREFETCH=1";; C) D=$P; E="L2_PREFETCH=0";; esac
    n=pf-d$dr-$r
    echo "$(date -Is) q26 $n"
    tools/budget_run.sh $n $D 0 BENCH_CASES=code,prose DRAFTS=$dr $X $E > logs/$n.out 2>&1
    echo "$(date -Is) q26 $n rc=$?"
  done
done
echo "$(date -Is) q26 done"
