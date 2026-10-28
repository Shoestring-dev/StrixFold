#!/bin/bash
# serve_run.sh NAME DEPLOY [KEY=VALUE...]: one timed served run on both boxes (after a-arch's serve_run.sh): conf NAME
# (mkconf.sh), DEPLOY's first_light.sh up (locks node0 then node1, probe, BENCH inside quiet windows), then down.
# Output: ~/tfp/run/serve/<RUN_ID>/ and ~/tfp/kdec/runs/NAME.log (with rank 0's done lines: tokens and SHA).
set -uo pipefail
NAME=$1 D=$2; shift 2
K=$HOME/tfp/kdec
$K/tools/mkconf.sh "$NAME" "$D" "$@" || exit 1
export TF_SERVE_CONF=$K/conf/$NAME.conf RUN_ID=kdec-$NAME-$(date -u +%H%M%S)
T=$HOME/tfp/$D/tools/dsv41-hip
{
  echo "== $(date -u +%FT%TZ) $NAME on $D ($RUN_ID) $*"
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
