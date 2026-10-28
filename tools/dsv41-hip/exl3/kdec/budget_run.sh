#!/bin/bash
# budget_run.sh NAME DEPLOY PROF(0|1) [KEY=VALUE...]: one served run on both boxes (kdec mkconf.sh, DEPLOY's
# first_light.sh up / down, as serve_run.sh) for the decode time budget. PROF=1: both ranks under rocprofv3 (kernel and
# memory-copy trace, CSV under ~/tfp/kdec/prof/NAME on each box). Always: the GPU clock every 1 ms on both boxes
# while the timed bench's quiet window is open (~/tfp/kdec/prof/NAME/sclk-<box>.jsonl). Log ~/tfp/kdec/runs/NAME.log.
# Run on node1, not under bg.sh (the ranks must not inherit its caps).
set -uo pipefail
NAME=$1 D=$2 PROF=$3; shift 3
K=$HOME/tfp/kdec
P=tfp/kdec/prof/$NAME
mkdir -p ~/$P; ssh -n node0 "mkdir -p ~/$P ~/tfp/kdec/tools"
scp -q $K/tools/sclk_sample.py node0:tfp/kdec/tools/
W=()
if [ "$PROF" = 1 ]; then
  W=("WRAP0='/opt/rocm/bin/rocprofv3 --kernel-trace --memory-copy-trace --output-format csv -d ~/$P --'"
     "WRAP1='/opt/rocm/bin/rocprofv3 --kernel-trace --memory-copy-trace --output-format csv -d ~/$P --'")
fi
$K/tools/mkconf.sh "$NAME" "$D" "$@" "${W[@]}" || exit 1
export TF_SERVE_CONF=$K/conf/$NAME.conf RUN_ID=kdec-$NAME-$(date -u +%H%M%S)
T=$HOME/tfp/$D/tools/dsv41-hip
# the samplers wait for the bench's quiet window (want.first-light-RUN_ID) and stop when it closes
ssh -n node0 "cd ~/$P && (exec taskset -c 31 python3 ~/tfp/kdec/tools/sclk_sample.py $RUN_ID ~/$P/sclk-node0.jsonl >sclk-node0.log 2>&1 </dev/null &)"
(cd ~/$P && exec taskset -c 31 python3 $K/tools/sclk_sample.py $RUN_ID ~/$P/sclk-node1.jsonl >sclk-node1.log 2>&1 </dev/null) &
{
  echo "== $(date -u +%FT%TZ) $NAME on $D ($RUN_ID) PROF=$PROF $*"
  $T/first_light.sh up; rc=$?
  echo "== up rc=$rc"
  $T/first_light.sh down
  echo "== rank 0 done lines"
  ssh -n node0 "grep -E 'done chatcmpl' ~/tfp/run/serve/$RUN_ID/rank0.log"
  echo "== bench"
  cat ~/tfp/run/serve/$RUN_ID/bench-rank0.txt 2>/dev/null
  echo "== warnings"
  ssh -n node0 "grep -E 'warning|error' ~/tfp/run/serve/$RUN_ID/rank0.log | tail -6"
  grep -E "warning|error" ~/tfp/run/serve/$RUN_ID/rank1.log | tail -4
} 2>&1 | tee $K/runs/$NAME.log
# the samplers end with the window; stop files for a window that never opened
touch ~/$P/sclk-node1.jsonl.stop; ssh -n node0 "touch ~/$P/sclk-node0.jsonl.stop"
wait
echo "== samplers"; cat ~/$P/sclk-node1.log; ssh -n node0 "cat ~/$P/sclk-node0.log"
echo "== traces"; find ~/$P -name "*kernel_trace.csv" | xargs -r ls -la; ssh -n node0 "find ~/$P -name '*kernel_trace.csv' | xargs -r ls -la"
