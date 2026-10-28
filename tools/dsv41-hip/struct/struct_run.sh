#!/bin/bash
# struct_run.sh CONF_BASE CONF_STRUCT RUN_ID [--drafts 0|1] [--cases LIST]: structured output served, one run under
# both GPU locks (node0's first; qual_lib.sh): the baseline build (CONF_BASE: the integration build this branch sits
# on) answers struct_serve.py's plain prompts, is stopped, then this branch's build (CONF_STRUCT) answers every case,
# its plain replies checked token for token against the baseline's. Both configurations are '$HOME/...' files present
# on both boxes. Writes ~/tfp/qual/out/RUN_ID-{base,struct}/ (driver.log, rank logs, struct-*.json) and
# ~/tfp/qual/out/RUN_ID-struct/struct-run.json; exits 0 only if every case passed. Nothing is left behind: ranks
# stopped and both locks released on every exit path. The locks are this script's own (ql_lock / ql_unlock below,
# pid files ~/tfp/struct/lock.pid on each box), not qual_lib's shared ~/tfp/qual/qlock.pid: two drivers sharing
# one pid file remove and kill each other's holders.
set -uo pipefail
CBASE=${1:?CONF_BASE}; CSTRUCT=${2:?CONF_STRUCT}; RUN=${3:?RUN_ID}; shift 3
DRAFTS=0; CASES=""
while [ $# -gt 0 ]; do
  case $1 in --drafts) DRAFTS=$2; shift 2;; --cases) CASES=$2; shift 2;; *) echo "unknown $1" >&2; exit 2;; esac
done
HERE=$(cd "$(dirname "$0")" && pwd)
QCONF=$CBASE QRUN=$RUN-base
# shellcheck source=../qual/qual_lib.sh
source "$HERE/../qual/qual_lib.sh"
ql_init
BASE_OUT=$QOUT
ssh -n node0 "mkdir -p ~/tfp/struct/bin" && scp -q "$HERE/struct_serve.py" node0:tfp/struct/bin/
mkdir -p "$HOME/tfp/struct"

# A holder per box: sh writes its own pid, then becomes flock, whose child marks the lock held and sleeps. Release
# kills the child (the lock goes with it) and the flock (a holder still waiting stops waiting), by pid.
hold_cmd() {  # hold_cmd WAIT_S: the holder's command line (paths under the box's own $HOME)
  echo "rm -f \$HOME/tfp/struct/lock.pid \$HOME/tfp/struct/lock.held; setsid -f sh -c 'echo \$\$ > \$HOME/tfp/struct/lock.pid; exec flock -w $1 /var/tmp/halo-gpu.lock sh -c \"touch \$HOME/tfp/struct/lock.held; exec sleep 7200\"' </dev/null >/dev/null 2>&1"
}
release_cmd='p=$(cat $HOME/tfp/struct/lock.pid 2>/dev/null); [ -n "$p" ] && { pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; }; rm -f $HOME/tfp/struct/lock.pid $HOME/tfp/struct/lock.held'

ql_lock() {
  ql_log "lock node0 (wait <= ${LOCK1_WAIT_S}s)"
  ssh -n node0 "$(hold_cmd "$LOCK1_WAIT_S"); for i in \$(seq 1 $LOCK1_WAIT_S); do [ -e ~/tfp/struct/lock.held ] && exit 0; sleep 1; done; exit 1" || { ql_log "node0 lock not free"; ql_unlock; return 1; }
  ql_log "lock node1 (wait <= ${LOCK2_WAIT_S}s)"
  bash -c "$(hold_cmd "$LOCK2_WAIT_S")"
  local i
  for i in $(seq 1 $((LOCK2_WAIT_S + 5))); do [ -e "$HOME/tfp/struct/lock.held" ] && { ql_log "both locks held"; return 0; }; sleep 1; done
  ql_log "node1 lock not free within ${LOCK2_WAIT_S}s: releasing both"
  ql_unlock
  return 1
}

ql_unlock() {
  bash -c "$release_cmd"
  ssh -n node0 "$release_cmd" || true
  ql_log "locks released"
}

cleanup() {
  set +e
  ql_stop
  ql_copy_logs
  ql_unlock
}
trap cleanup EXIT

serve() {  # serve TAG ARGS...: struct_serve.py on node0, its JSON copied to QOUT
  local tag=$1; shift
  ssh -n node0 "python3 ~/tfp/struct/bin/struct_serve.py http://127.0.0.1:$QPORT $QRUN0/struct-$tag.json --model $QNAME $*" | tee -a "$QOUT/driver.log"
  local rc=${PIPESTATUS[0]}
  scp -q "node0:$QRUN0/struct-$tag.json" "$QOUT/" || true
  return "$rc"
}

echo "{\"node0\": \"$(ssh -n node0 cat /proc/loadavg)\", \"node1\": \"$(cat /proc/loadavg)\"}" > "$QOUT/load-before.json"
ql_lock || exit 3
ql_start || { ql_log "the baseline server did not come up"; exit 1; }
ql_phase base
serve plain --cases plain; base_rc=$?
ql_stop
ql_copy_logs

# this branch's build, the same locks held
QCONF=$CSTRUCT QRUN=$RUN-struct
ql_init
ql_start || { ql_log "the struct server did not come up"; exit 1; }
ql_phase struct
scp -q "$BASE_OUT/struct-plain.json" "node0:$QRUN0/reference.json"
serve all --drafts "$QDRAFTS" --reference "$QRUN0/reference.json" ${CASES:+--cases $CASES}; struct_rc=$?
ssh -n node0 "grep -h 'structured reply' $QRUN0/rank0.log" > "$QOUT/grammar-cost.txt" 2>/dev/null || true
ql_phase done
cleanup
trap - EXIT
echo "{\"node0\": \"$(ssh -n node0 cat /proc/loadavg)\", \"node1\": \"$(cat /proc/loadavg)\"}" > "$QOUT/load-after.json"
python3 - "$QOUT" "$base_rc" "$struct_rc" <<'PY'
import json, sys
out, base_rc, struct_rc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
doc = json.load(open(f"{out}/struct-all.json"))
verdict = {"passed": base_rc == 0 and struct_rc == 0 and doc.get("passed"), "base_rc": base_rc, "struct_rc": struct_rc,
           "cases": {k: {"ok": v["ok"], "detail": v["detail"]} for k, v in doc["cases"].items()}}
json.dump(verdict, open(f"{out}/struct-run.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(verdict, ensure_ascii=False)[:4000])
sys.exit(0 if verdict["passed"] else 1)
PY
