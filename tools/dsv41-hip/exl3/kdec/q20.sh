#!/bin/bash
# q20 (k-dec2): after the lead's overnight unit: the clock calibration and the MALL test (node1), then the decode
# budget's served runs (both boxes: drafts off / on, with and without rocprofv3). One GPU lock (or pair) a run. Run as
# a plain user unit (no caps: the ranks start here).
cd ~/tfp/kdec
log() { echo "$(date -Is) q20 $*"; }
while systemctl --user is-active -q overnight-b010; do sleep 30; done
log "overnight unit done"
mkdir -p prof/clockcal
tools/runq.sh clockcal /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv -d $HOME/tfp/kdec/prof/clockcal -- $HOME/tfp/kdec/hip/clockcal
tools/runq.sh mall hip/mall
D=deploy/b010ee4ae317
S="BENCH_CASES=code,prose"
for x in n-off:0:0 p-off:1:0 p-on:1:1 n-on:0:1; do
  IFS=: read -r t p dr <<< "$x"
  log "served bud-$t"
  tools/budget_run.sh bud-$t $D $p $S DRAFTS=$dr > logs/bud-$t.out 2>&1
  log "served bud-$t rc=$?"
done
log done
