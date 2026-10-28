#!/bin/bash
# lane_gate.sh CONF TAG REQUESTS [RANK0_ARG...]: one two-rank launch of the lane gate (tf-dsv41-lanes: rank 0 on node0
# serving REQUESTS, rank 1 on node1 following) with configuration CONF ('$HOME/...', the same file on both boxes), each
# rank from the configuration's deployment through its qual/lane_rank.sh: the environment and engine flags are the
# served ones (serve_lib.sh), so --pool, --streams, --drafts, --side, --aio, TF_GPU_STRICT, TF_DS_AUX and
# TF_DS_REPLAY_FLOOR come from CONF. Rank 0 also gets --requests, --parallel PARALLEL, --logits-sha 1 and RANK0_ARGs;
# LANE_DUMP=1 adds --logits-dump (each prompt's last-row logits; copied back to OUT/logits).
# One run under both GPU locks (qual_lib.sh: node0's first, all or nothing), released on every exit path, both ranks
# stopped by pid; nothing left behind. Run on node1.
# OUT = ~/tfp/qual/out/lane-TAG: both ranks' stdout/stderr, commands, resolved configurations, the requests file, the
# boxes' load and sclk before and after, and lane.json (a_grade.py lane: every request's rows, logits sha256, reply
# token_sha, errors, fallbacks). Exit 0 when rank 0 served every request and rank 1 followed to the end.
set -uo pipefail
QCONF=${1:?CONF}; TAG=${2:?TAG}; REQ=${3:?REQUESTS}; shift 3
QRUN=lane-$TAG
QLOCK_MAX_S=${QLOCK_MAX_S:-3000}        # the run's own length at most (load ~2 min each + the requests)
LANE_TIMEOUT_S=${LANE_TIMEOUT_S:-2700}
source "$(dirname "$0")/qual_lib.sh"
ql_init
[ -f "$REQ" ] || { echo "no $REQ" >&2; exit 2; }
PAR=$( (source "$QCONF1"; echo "$PARALLEL") )
cp "$REQ" "$QOUT/requests.json"
scp -q "$REQ" "node0:$QRUN0/requests.json"
ssh -n node0 "rm -f $QRUN0/rank0.pid $QRUN0/rank0.jsonl $QRUN0/rank0.err; rm -rf $QRUN0/logits"
rm -f "$QRUN1/rank1.pid" "$QRUN1/rank1.jsonl" "$QRUN1/rank1.err"
ARGS0=(--requests "$QRUN0/requests.json" --parallel "$PAR" --logits-sha 1)
[ "${LANE_DUMP:-0}" = 1 ] && ARGS0+=(--logits-dump "$QRUN0/logits")
ARGS0+=("$@")

pid0() { ssh -n node0 "f=$QRUN0/rank0.pid; [ -s \$f ] && kill -0 \$(cat \$f) 2>/dev/null && cat \$f" 2>/dev/null || true; }
pid1() { [ -s "$QRUN1/rank1.pid" ] && kill -0 "$(cat "$QRUN1/rank1.pid")" 2>/dev/null && cat "$QRUN1/rank1.pid"; return 0; }
state() { echo "$1 node0 $(ssh -n node0 'cat /proc/loadavg; cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep "\*" | tr "\n" " "') /" \
  "node1 $(cat /proc/loadavg; cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep '\*' | tr '\n' ' ')" >> "$QOUT/load.txt"; }
cleanup() {
  set +e
  local p
  p=$(pid0); [ -n "$p" ] && { ql_log "stopping rank 0 ($p)"; ssh -n node0 "kill $p"; }
  p=$(pid1); [ -n "$p" ] && { ql_log "stopping rank 1 ($p)"; kill "$p"; }
  sleep 3
  p=$(pid0); [ -n "$p" ] && ssh -n node0 "kill -9 $p"
  p=$(pid1); [ -n "$p" ] && kill -9 "$p"
  ql_unlock
}
trap cleanup EXIT
ql_lock || exit 3
state before
ql_log "lane gate $TAG: conf $QCONF, deploy $QDEPLOY, rank 0 args ${ARGS0[*]}"
t0=$(date +%s)
setsid -f "$QT1/qual/lane_rank.sh" "$QCONF1" 1 "$QRUN1" </dev/null > "$QOUT/lane_rank1.out" 2>&1
sleep 1
ssh -n node0 "setsid -f $QT0/qual/lane_rank.sh $QCONF0 0 $QRUN0 ${ARGS0[*]} </dev/null > $QRUN0/lane_rank0.out 2>&1"
sleep 5
[ -n "$(pid0)" ] || { ql_log "rank 0 did not start: $(ssh -n node0 "cat $QRUN0/lane_rank0.out; tail -3 $QRUN0/rank0.err" 2>/dev/null)"; }
end=$((t0 + LANE_TIMEOUT_S))
while [ -n "$(pid0)" ] && [ "$(date +%s)" -lt "$end" ]; do sleep 5; done
timed_out=0; [ -n "$(pid0)" ] && { timed_out=1; ql_log "rank 0 still running after ${LANE_TIMEOUT_S}s: stopped"; }
for _ in $(seq 1 60); do [ -z "$(pid1)" ] && break; sleep 2; done
secs=$(( $(date +%s) - t0 ))
cleanup   # (anything still running) and the locks
trap - EXIT
state after
ql_log "lane gate $TAG done in ${secs}s (timed out: $timed_out)"
scp -q "node0:$QRUN0/rank0.jsonl" "node0:$QRUN0/rank0.err" "node0:$QRUN0/rank0.cmd" "node0:$QRUN0/resolved-rank0.json" \
  "node0:$QRUN0/lane_rank0.out" "$QOUT/" 2>/dev/null
ssh -n node0 "test -d $QRUN0/logits" && scp -q -r "node0:$QRUN0/logits" "$QOUT/"
cp "$QRUN1/rank1.jsonl" "$QRUN1/rank1.err" "$QRUN1/rank1.cmd" "$QRUN1/resolved-rank1.json" "$QOUT/" 2>/dev/null
python3 "$QTOOLS/a_grade.py" lane "$QOUT" --tag "$TAG" --conf "$QCONF1" --seconds "$secs" --timed-out "$timed_out"
