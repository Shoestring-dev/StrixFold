#!/bin/bash
# first_light.sh up | down | restart | status: the TP2 server on both boxes from serve.conf, run on node1 (rank 1's box;
# it reaches node0 by ssh). Both boxes run one deployed commit (deploy_rank0.sh --native, to node0 and --host local);
# rank 0 uses node0's deployed copy of these tools (DEPLOY_DIR/tools/dsv41-hip), rank 1 this box's. TF_SERVE_CONF, when
# set, must name a file at the same place under each box's $HOME.
#   up       a run (RUN_ID, default the UTC time) on both boxes:
#            1. both ranks resolved and checked (check): the same deploy commit, kit manifest and config on both
#            2. the GPU locks, all or nothing: node0's (waiting up to LOCK_WAIT_S), then node1's within
#               LOCK_PAIR_S; if node1's is busy, node0's is released and the pair retried after 30 s (LOCK_TRIES times);
#               held for the run
#            3. rank 1 started (it loads, then waits for rank 0's link), then rank 0
#            4. ready: rank 0's /health and the probe chat request answered (PROBE_EXPECT in the reply), rank 1 linked
#            5. BENCH=1: the timed request inside quiet windows (the optional QUIET_BIN hook on: node0, then node1;
#               off on both after):
#               TTFT, prefill and decode tok/s, the server's /metrics, load on both boxes before and after
#            RUN/first_light.json: both resolved configurations, the probe, the timed run, uptime on both boxes.
#            Any failure takes the run down again (exit 1).
#   down     rank 0 stopped cleanly (SIGTERM: the engine closes, rank 1 hears done and leaves), rank 1 stopped if it
#            has not left, both locks released
#   restart  down, then up (a new run)
#   status   both ranks
set -Eeuo pipefail  # (-E: the ERR trap reaches failures inside functions)
cd "$(dirname "$0")"
source ./serve_lib.sh
serve_load 1
R0=$RANK0_HOST
R0_TOOLS=$DEPLOY_DIR_RAW/tools/dsv41-hip   # expanded on node0 (its $HOME)

# TF_SERVE_CONF under this box's $HOME names the same file under node0's (the two homes can differ)
R0_CONF=${TF_SERVE_CONF:+${TF_SERVE_CONF/#$HOME/\$HOME}}
r0() { ssh "$R0" "RUN_ID=$RUN_ID${R0_CONF:+ TF_SERVE_CONF=$R0_CONF} $R0_TOOLS/serve-rank0.sh $*"; }

# quiet windows: an optional hook that holds other work on a box while a timed run goes (QUIET_BIN, default
# ~/bin/halo-quiet, called as QUIET_BIN on|off TAG; skipped on a box where it isn't installed). node0 first on, both off.
quiet_both() {
  local q='b=${QUIET_BIN:-$HOME/bin/halo-quiet}; [ -x "$b" ] && "$b"'
  if [ "$1" = on ]; then ssh "$R0" "$q on $QUIET_TAG || true"; eval "$q on \"\$QUIET_TAG\" || true"
  else eval "$q off \"\$QUIET_TAG\" || true"; ssh "$R0" "$q off $QUIET_TAG || true"; fi
}

down() {
  set +e
  # a quiet window left on by a failure (the requests also expire after 20 minutes)
  [ -n "${QUIET_TAG:-}" ] && quiet_both off
  r0 stop
  local i
  for i in $(seq 1 "$STOP_TIMEOUT_S"); do [ -z "$(serve_pid "$RUN/rank1.pid")" ] && break; sleep 1; done
  ./serve-rank1.sh stop
  ./serve-rank1.sh unlock
  r0 unlock
  set -e
}

up() {
  RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
  export RUN_ID
  RUN=$RUN_ROOT/$RUN_ID
  mkdir -p "$RUN"
  echo "run $RUN_ID: rank 0 on $R0, rank 1 on $(hostname)"
  r0 check > "$RUN/resolved-rank0.json"
  ./serve-rank1.sh check > /dev/null
  python3 - "$RUN/resolved-rank0.json" "$RUN/resolved-rank1.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for k in ("deploy", "kit_manifest_sha256", "config_sha256"):
    x, y = (a[k]["commit"], b[k]["commit"]) if k == "deploy" else (a[k], b[k])
    if x != y:
        sys.exit(f"the ranks differ in {k}: rank 0 {x}, rank 1 {y}")
# the transport by name: one config hash already says so, but a mixed pair (one rank's TF_COLL_KIND tcp proxy or
# host, the other's rdma) is named here before any rank starts, and the engines' handshake refuses it too (kind=)
ea, eb = a["env"], b["env"]
for k in ("TF_COLL_KIND", "TF_TP_STRIPE", "TF_RDMA_DEVICES", "TF_RDMA_IFNAME", "TF_RDMA_GID_INDEX"):
    if ea.get(k) != eb.get(k):
        sys.exit(f"the ranks differ in {k}: rank 0 {ea.get(k, '(unset)')}, rank 1 {eb.get(k, '(unset)')}"
                 + (" (a tcp/rdma mixed pair)" if k == "TF_COLL_KIND" else ""))
kind = ea.get("TF_COLL_KIND", "host")
wire = f"rdma (devices {ea.get('TF_RDMA_DEVICES') or ea.get('TF_RDMA_IFNAME') or 'resolved'}, gid {ea.get('TF_RDMA_GID_INDEX', '3')})" if kind == "rdma" else f"tcp ({kind})"
print(f"both ranks: deploy {a['deploy']['commit'][:12]}, kit {a['kit_manifest_sha256'][:12]}, config {a['config_sha256'][:12]}, collectives {wire}")
PY
  trap 'echo "first light failed: taking run $RUN_ID down" >&2; down; exit 1' ERR
  local try
  for try in $(seq 1 "${LOCK_TRIES:-20}"); do
    r0 lock
    ./serve-rank1.sh lock "${LOCK_PAIR_S:-60}" && break
    r0 unlock
    [ "$try" -lt "${LOCK_TRIES:-20}" ] || { trap - ERR; die "node1's GPU lock stayed busy (${LOCK_TRIES:-20} tries)"; }
    sleep 30
  done
  ./serve-rank1.sh start --locked
  r0 start --locked
  # (not a pipe: a failure in a pipeline's subshell would take the run down twice)
  r0 ready > "$RUN/rank0-ready.txt" 2>&1 || { cat "$RUN/rank0-ready.txt"; false; }
  cat "$RUN/rank0-ready.txt"
  ./serve-rank1.sh wait-linked
  trap - ERR
  grep -m1 '^{' "$RUN/rank0-ready.txt" > "$RUN/probe.json"
  if [ "$BENCH" = 1 ]; then
    # quiet windows on both boxes (node0 first), always paired with off
    QUIET_TAG=first-light-$RUN_ID
    quiet_both on
    { echo "{\"rank0\": \"$(ssh "$R0" uptime)\", \"rank1\": \"$(uptime)\"}"; } > "$RUN/load-before.json"
    r0 bench > "$RUN/bench-rank0.txt" 2>&1 || echo "the timed run failed (RUN/bench-rank0.txt)" >&2
    { echo "{\"rank0\": \"$(ssh "$R0" uptime)\", \"rank1\": \"$(uptime)\"}"; } > "$RUN/load-after.json"
    quiet_both off; QUIET_TAG=
    grep -m1 '^{' "$RUN/bench-rank0.txt" > "$RUN/bench.json" || true
    cat "$RUN/bench-rank0.txt"
  fi
  python3 - "$RUN" "$RUN_ID" "$(ssh "$R0" uptime)" "$(uptime)" <<'PY'
import json, os, sys
run, run_id, up0, up1 = sys.argv[1:5]
j = lambda n: json.load(open(os.path.join(run, n))) if os.path.exists(os.path.join(run, n)) else None
json.dump({"schema": "tf-port-first-light/1", "run": run_id, "rank0": j("resolved-rank0.json"), "rank1": j("resolved-rank1.json"),
           "probe": j("probe.json"), "bench": j("bench.json"), "load_before_bench": j("load-before.json"),
           "load_after_bench": j("load-after.json"), "uptime": {"rank0": up0, "rank1": up1}},
          open(os.path.join(run, "first_light.json"), "w"), indent=1)
PY
  echo "ready: run $RUN_ID ($RUN/first_light.json)"
}

case ${1:-} in
  up) up;;
  down)
    RUN_ID=${RUN_ID:-$(serve_latest)}; [ -n "$RUN_ID" ] || die "no run"; RUN=$RUN_ROOT/$RUN_ID; down;;
  restart)
    RUN_ID=${RUN_ID:-$(serve_latest)}
    if [ -n "$RUN_ID" ]; then RUN=$RUN_ROOT/$RUN_ID; down; fi
    unset RUN_ID; up;;
  status)
    RUN_ID=${RUN_ID:-$(serve_latest)}; RUN=$RUN_ROOT/${RUN_ID:-none}
    r0 status || true; ./serve-rank1.sh status;;
  *) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2;;
esac
