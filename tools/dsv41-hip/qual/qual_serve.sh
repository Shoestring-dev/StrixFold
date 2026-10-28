#!/bin/bash
# qual_serve.sh CONF RUN_ID [--long 0|1] [--cases LIST] [--restart 0|1]: the served qualification block (one run under
# both GPU locks, node0's first): both ranks from the configuration CONF ('$HOME/...', the same file on both boxes),
# then on the live server
#   1. warmup probe, then the shared objects mapped by both ranks (no NVIDIA/CUDA library; libamdhip64 present)
#   2. serve_qual.py's functional suite on rank 0's box (serve_qual.json), then (--struct 1, the default, a full suite
#      only) s-struct's struct_serve.py: structured output and required tool calls (serve_struct.json)
#   3. a short idle (steady state) and the server's /metrics
#   4. --restart 1 (default): both ranks stopped cleanly and started again; the reference requests (probe, repeat)
#      must give the first start's token ids (clean restart, handoff M5)
# while memledger.py samples both boxes once a second (allocation/residency ledger, handoff section 5).
# Writes ~/tfp/qual/out/RUN_ID/qual-serve.json (the verdict, with every evidence file) and exits 0 only if all passed.
# Nothing is left behind: ranks stopped, samplers stopped, both locks released on every exit path. Exit 4: rank 0
# refused to load because node0 was not idle (U25): no result, run it again.
set -uo pipefail
QCONF=${1:?CONF}; QRUN=${2:?RUN_ID}; shift 2
LONG=0; CASES=""; RESTART=1; STRUCT=1
while [ $# -gt 0 ]; do
  case $1 in --long) LONG=$2; shift 2;; --cases) CASES=$2; shift 2;; --restart) RESTART=$2; shift 2;; --struct) STRUCT=$2; shift 2;;
    *) echo "unknown $1" >&2; exit 2;; esac
done
QLOCK_MAX_S=${QLOCK_MAX_S:-$([ "$LONG" = 1 ] && echo 2700 || echo 1500)}   # two loads, the suite, the restart
source "$(dirname "$0")/qual_lib.sh"
ql_init
PLAN_GIB=${PLAN_GIB:-104.6}   # memory_plan.md section 4: the device plan's upper bound per rank

cleanup() {
  set +e
  ql_stop
  ql_sampler_stop
  ql_copy_logs
  ql_unlock
}
trap cleanup EXIT

suite() {  # suite TAG CASES: serve_qual.py on node0, result copied to QOUT/serve_qual-TAG.json
  local args="--model $QNAME --context $QCTX --drafts $QDRAFTS --long $LONG"
  [ -n "$2" ] && args="$args --cases $2"
  [ -n "${3:-}" ] && args="$args $3"
  ssh -n node0 "python3 ~/tfp/qual/bin/serve_qual.py http://127.0.0.1:$QPORT $QRUN0/serve_qual-$1.json $args" | tee -a "$QOUT/driver.log"
  local rc=${PIPESTATUS[0]}
  scp -q "node0:$QRUN0/serve_qual-$1.json" "$QOUT/" || true
  return "$rc"
}

ql_log "run $QRUN: conf $QCONF, deploy $QDEPLOY, context $QCTX, drafts $QDRAFTS, graphs $QGRAPHS"
echo "{\"node0\": \"$(ssh -n node0 cat /proc/loadavg)\", \"node1\": \"$(cat /proc/loadavg)\"}" > "$QOUT/load-before.json"
ql_lock || exit 3
ql_sampler_start
start_t=$(date +%s)
if ! ql_start; then
  ql_copy_logs
  if ql_env_refusal; then ql_log "node0 not idle: rank 0 refused to load (memory budget, U25); nothing measured"; exit 4; fi
  ql_log "the server did not come up"; exit 1
fi
load_s=$(( $(date +%s) - start_t ))

ql_phase warmup
suite warmup probe > /dev/null
ql_maps "$QOUT/maps.txt"; maps_rc=$?
ql_log "maps check rc=$maps_rc"

ql_phase suite
suite main "$CASES"; suite_rc=$?

# structured output and required tool calls (s-struct's struct_serve.py, U44/U45), when this checkout has it
struct_rc=
SS=$QTOOLS/../struct/struct_serve.py
if [ "$STRUCT" = 1 ] && [ -f "$SS" ] && [ -z "$CASES" ]; then
  ql_phase struct
  scp -q "$SS" node0:tfp/qual/bin/
  ssh -n node0 "python3 ~/tfp/qual/bin/struct_serve.py http://127.0.0.1:$QPORT $QRUN0/serve_struct.json --model $QNAME --drafts $QDRAFTS" | tee -a "$QOUT/driver.log"
  struct_rc=${PIPESTATUS[0]}
  scp -q "node0:$QRUN0/serve_struct.json" "$QOUT/" || true
fi

ql_phase steady
sleep 15
ssh -n node0 "curl -fsS -m 10 http://127.0.0.1:$QPORT/metrics" > "$QOUT/metrics.txt" 2>/dev/null || true

restart_rc=0
if [ "$RESTART" = 1 ]; then
  ql_stop
  ql_copy_logs
  for f in rank0.log rank1.log; do [ -f "$QOUT/$f" ] && mv "$QOUT/$f" "$QOUT/first-$f"; done
  ql_log "restart"
  restart_t=$(date +%s)
  ql_start || { ql_log "the restart did not come up"; restart_rc=1; }
  restart_s=$(( $(date +%s) - restart_t ))
  if [ $restart_rc = 0 ]; then
    ql_phase restart
    suite restart probe,repeat || restart_rc=1
  fi
fi
ql_phase done
cleanup
trap - EXIT

python3 "$QTOOLS/run_manifest.py" "$QOUT/manifest.json" --deploy "${QDEPLOY//\$HOME/$HOME}" --conf "$QCONF1" \
  --resolved "$QOUT/resolved-rank0.json" --resolved "$QOUT/resolved-rank1.json" --note "qual_serve.sh $QRUN"
python3 "$QTOOLS/memledger.py" summary "$QOUT/mem-node0.jsonl" "$PLAN_GIB" > "$QOUT/mem-node0.summary.json"; m0=$?
python3 "$QTOOLS/memledger.py" summary "$QOUT/mem-node1.jsonl" "$PLAN_GIB" > "$QOUT/mem-node1.summary.json"; m1=$?
echo "{\"node0\": \"$(ssh -n node0 cat /proc/loadavg)\", \"node1\": \"$(cat /proc/loadavg)\"}" > "$QOUT/load-after.json"

python3 "$QTOOLS/serve_verdict.py" "$QOUT" "$QCONF1" "$maps_rc" "$suite_rc" "$RESTART" "$restart_rc" "$m0" "$m1" "${load_s:-}" "${restart_s:-}" "${struct_rc:-}"
