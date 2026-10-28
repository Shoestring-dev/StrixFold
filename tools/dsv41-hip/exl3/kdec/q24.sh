#!/bin/bash
# q24 (k-dec2): the HIP graph-scheduling knob alone, drafts off, served on deploy/current (b010ee4), A B B A:
# A = default, B = DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1 on both ranks (through WRAP0 / WRAP1: /usr/bin/env VAR=1).
# Plain user unit (the ranks start here).
cd ~/tfp/kdec
S="BENCH_CASES=code,prose DRAFTS=0"
for x in A1 B1 B2 A2; do
  echo "$(date -Is) q24 $x"
  if [ ${x:0:1} = B ]; then
    tools/budget_run.sh seg-$x deploy/current 0 $S "WRAP0='/usr/bin/env DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1'" "WRAP1='/usr/bin/env DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1'" > logs/seg-$x.out 2>&1
  else
    tools/budget_run.sh seg-$x deploy/current 0 $S > logs/seg-$x.out 2>&1
  fi
  echo "$(date -Is) q24 $x rc=$?"
done
echo "$(date -Is) q24 done"
