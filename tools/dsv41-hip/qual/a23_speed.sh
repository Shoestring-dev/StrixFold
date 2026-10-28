#!/bin/bash
# a23_speed.sh CONF_DIR SESSION: the A2+A3 speed session of PLAN-research-actions.md section 4: 5 rounds of off, side,
# aio (CONF_DIR/a23-{off,side,aio}.conf), then the closing pair off and side+aio (a23-both.conf): 17 launches, each its
# own a_served.sh speed run (both locks per launch, released between launches so other runs interleave; a warm-up 8K
# prefill discarded; the code and prose decodes at 400 tokens; one 8K prefill), every 8K document cut at its own offset
# of long_memory_archive.txt (541 characters apart: no document is used twice in the session). CONF_DIR is '$HOME/...'.
# SESSION: the session's name; ~/tfp/qual/out/SESSION/session.jsonl lists the launches (round, arm, dir) as they finish,
# then a_grade.py speed applies the rule. A launch that fails stops the session (reported, never skipped).
set -uo pipefail
CD=${1:?CONF_DIR}; S=${2:?SESSION}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${QOUT_ROOT:-$HOME/tfp/qual/out}
mkdir -p "$ROOT/$S"
L=$ROOT/$S/session.jsonl
k=0
run() {  # run ROUND ARM CONF
  local tag=$S-r$1-$2
  if QOUT_ROOT=$ROOT "$HERE/a_served.sh" "$CD/$3.conf" "$tag" speed --warm-offset $((2 * k * 541)) --offset $(((2 * k + 1) * 541)); then
    echo "{\"round\": $1, \"arm\": \"$2\", \"dir\": \"$ROOT/served-$tag\"}" >> "$L"
  else
    echo "$(date -u +%H:%M:%S) launch $tag failed: session stopped" | tee -a "$ROOT/$S/session.log"; exit 1
  fi
  k=$((k + 1))
}
for r in 1 2 3 4 5; do
  run "$r" off a23-off
  run "$r" side a23-side
  run "$r" aio a23-aio
done
run 6 off_close a23-off
run 6 both_close a23-both
python3 "$HERE/a_grade.py" speed --session "$L"
