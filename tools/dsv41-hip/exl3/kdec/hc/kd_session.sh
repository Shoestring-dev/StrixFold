#!/bin/bash
# kd_session.sh NAME DEPLOY DRAFTS EST_ROUNDS: one server (both boxes, TF_HC_FUSE=1 through WRAP0 / WRAP1) and a long
# hc_session.py run against it from node0; then rank 0's done lines: rounds in all, and the code / prose replies' SHAs
# before and after (must be equal). Run on node1 as a plain user unit.
set -uo pipefail
NAME=$1 D=$2 DR=$3 EST=$4
K=$HOME/tfp/kdec
$K/tools/mkconf.sh $NAME $D BENCH=0 DRAFTS=$DR "WRAP0='/usr/bin/env TF_HC_FUSE=1'" "WRAP1='/usr/bin/env TF_HC_FUSE=1'" >/dev/null || exit 1
export TF_SERVE_CONF=$K/conf/$NAME.conf RUN_ID=kdec-$NAME-$(date -u +%H%M%S)
T=$HOME/tfp/$D/tools/dsv41-hip
scp -q $K/hc/hc_session.py node0:tfp/kdec/
{
  echo "== $(date -u +%FT%TZ) $NAME on $D drafts $DR ($RUN_ID)"
  $T/first_light.sh up; echo "== up rc=$?"
  ssh -n node0 "python3 ~/tfp/kdec/hc_session.py http://127.0.0.1:18080 ~/tfp/kdec/$NAME-session.json $EST"
  $T/first_light.sh down
  ssh -n node0 "grep -E 'done chatcmpl|cancel|error|warning' ~/tfp/run/serve/$RUN_ID/rank0.log" > $K/runs/$NAME-done.txt
  python3 - $K/runs/$NAME-done.txt <<'PY'
import re, sys
L = [l for l in open(sys.argv[1]) if "done chatcmpl" in l]
rounds = sum(int(re.search(r"rounds=(\d+)", l).group(1)) for l in L)
shas = [re.search(r"tokens=(\d+) sha=(\w+) finish=(\w+)", l).groups() for l in L]
print("requests", len(L), "rounds in all", rounds)
for s in shas: print("  ", s)
pre, post = shas[1:3], shas[-2:]
print("code / prose before", pre, "after", post, "EQUAL" if pre == post else "DIFFERENT")
PY
  grep -vE "done chatcmpl" $K/runs/$NAME-done.txt | tail -5
} 2>&1 | tee $K/runs/$NAME.log
