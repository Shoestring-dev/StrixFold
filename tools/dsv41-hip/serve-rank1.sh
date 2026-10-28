#!/bin/bash
# serve-rank1.sh start [--locked] | stop | status | wait-linked: rank 1 of the TP2 server (node1), tf-dsv41-lanes
# following rank 0, from serve.conf (serve_lib.sh resolves it; RUN_ID names the run, default the UTC time).
#   start        resolve, check the inputs, write RUN/resolved-rank1.json, start rank 1 in the background
#                (RUN/rank1.log, RUN/rank1.pid) under this box's GPU lock (--locked: the caller holds it already)
#   wait-linked  until rank 1 has loaded and linked to rank 0 (its "loaded_s" line), or it exits, or READY_TIMEOUT_S
#   stop         SIGTERM (rank 1 also leaves by itself when rank 0 stops), SIGKILL after STOP_TIMEOUT_S
#   check   resolve and check only: RUN/resolved-rank1.json printed (identities: deploy commit, kit, config)
#   lock [WAIT_S] / unlock   hold (waiting at most WAIT_S, default LOCK_WAIT_S) / release this box's GPU lock for
#           the run (first_light.sh takes node0's, then node1's)
#   status       running or not, and the log's last line
set -euo pipefail
source "$(dirname "$0")/serve_lib.sh"
serve_load 1
RUN_ID=${RUN_ID:-$(serve_latest)}
RUN=$RUN_ROOT/${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
case ${1:-} in
  start)
    mkdir -p "$RUN"; ln -sfn "$(basename "$RUN")" "$RUN_ROOT/latest"
    [ -z "$(serve_pid "$RUN/rank1.pid")" ] || die "rank 1 already runs ($(cat "$RUN/rank1.pid"))"
    serve_check; serve_idle; serve_env; serve_cmd; serve_resolved "$RUN/resolved-rank1.json"
    serve_spawn "$RUN/rank1.pid" "$RUN/rank1.log" "$([ "${2:-}" = --locked ] && echo 1 || echo 0)"
    echo "rank 1 started ($(cat "$RUN/rank1.pid")): $RUN";;
  wait-linked)
    for _ in $(seq 1 "$READY_TIMEOUT_S"); do
      grep -q '"loaded_s"' "$RUN/rank1.log" 2>/dev/null && { echo "rank 1 linked: $(grep -m1 '"loaded_s"' "$RUN/rank1.log")"; exit 0; }
      [ -n "$(serve_pid "$RUN/rank1.pid")" ] || { echo "rank 1 exited:"; tail -5 "$RUN/rank1.log"; exit 1; }
      sleep 1
    done
    die "rank 1 not linked within ${READY_TIMEOUT_S}s";;
  check) mkdir -p "$RUN"; serve_check; serve_env; serve_cmd; serve_resolved "$RUN/resolved-rank1.json"; cat "$RUN/resolved-rank1.json";;
  lock) mkdir -p "$RUN"; serve_lock_hold "$RUN" "${2:-}"; echo "GPU lock held on $(hostname) ($(cat "$RUN/lock.pid"))";;
  unlock) serve_lock_release "$RUN"; echo "GPU lock released on $(hostname)";;
  stop) serve_stop "$RUN/rank1.pid" "rank 1";;
  status)
    pid=$(serve_pid "$RUN/rank1.pid"); echo "rank 1 ${pid:+running ($pid)}${pid:-not running}: $RUN"
    tail -1 "$RUN/rank1.log" 2>/dev/null || true;;
  *) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2;;
esac
