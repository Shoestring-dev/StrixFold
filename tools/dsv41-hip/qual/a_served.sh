#!/bin/bash
# a_served.sh CONF TAG MODE [a_served.py options]: one served launch for the parity checks (PLAN-research-actions.md
# sections 1 and 4): both GPU locks (qual_lib.sh: node0's first, all or nothing), both ranks from CONF's deployment
# (serve-rank0.sh / serve-rank1.sh, --locked) until /health answers, a_served.py MODE run on node0 against rank 0, both
# ranks stopped, the locks released on every exit path; nothing left behind. Run on node1.
#   a1     --requests FILE (on this box; copied to node0): A1's served check
#   a4     A4's sampling checks
#   speed  --warm-offset W --offset M: one launch of the A2/A3 speed session. Before the requests: the Engram shards'
#          page-cache residency on both boxes (residency.py), then quiet windows (node0, then node1) around the
#          requests; load and sclk before and after on both boxes.
# OUT = ~/tfp/qual/out/served-TAG (QOUT_ROOT moves it): served.json (a_served.py), both ranks' logs and resolved
# configurations, residency.json, load.txt, driver.log.
set -uo pipefail
QCONF=${1:?CONF}; TAG=${2:?TAG}; MODE=${3:?MODE}; shift 3
QRUN=served-$TAG
QLOCK_MAX_S=${QLOCK_MAX_S:-1800}
source "$(dirname "$0")/qual_lib.sh"
ql_init
ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --requests) cp "$2" "$QOUT/requests.json"; scp -q "$2" "node0:$QRUN0/requests.json"; ARGS+=(--requests "$QRUN0/requests.json"); shift 2;;
    *) ARGS+=("$1"); shift;;
  esac
done
ENGRAM=$( (source "$QCONF1"; echo "$ENGRAM_DIR") )
SHARDS="model-00047-of-00048.safetensors model-00048-of-00048.safetensors"
if [ "$MODE" = speed ]; then
  scp -q "$HOME/tfp/par/texts/long_memory_archive.txt" "node0:$QRUN0/long_memory_archive.txt"
  ARGS+=(--texts "$QRUN0/long_memory_archive.txt")
fi
state() { echo "$1 node0 $(ssh -n node0 'cat /proc/loadavg; grep "\*" /sys/class/drm/card0/device/pp_dpm_sclk | tr "\n" " "') /" \
  "node1 $(cat /proc/loadavg; grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | tr '\n' ' ')" >> "$QOUT/load.txt"; }
quiet() { ql_quiet "$1" "$QRUN" >> "$QOUT/driver.log"; }  # quiet on|off: node0 first (qual_lib.sh)
cleanup() { set +e; quiet off; ql_stop; ql_copy_logs; ql_unlock; }
trap cleanup EXIT
ql_lock || exit 3
state locked
ql_start || { ql_log "start failed"; ql_env_refusal && ql_log "rank 0 refused: memory not free (unrun)"; exit 1; }
ql_log "ready: $MODE ${ARGS[*]}"
if [ "$MODE" = speed ]; then
  d=${ENGRAM//\$HOME/}
  r1=$(cd "$HOME/$d" 2>/dev/null && python3 "$QTOOLS/residency.py" $SHARDS)
  r0=$(ssh -n node0 "cd \$HOME/$d && python3 $QT0/qual/residency.py $SHARDS")
  echo "{\"node0\": $r0, \"node1\": $r1}" > "$QOUT/residency.json"
  quiet on
  sleep 3
fi
state before
ssh -n node0 "python3 $QT0/qual/a_served.py $MODE http://127.0.0.1:$QPORT $QRUN0/served.json --model $QNAME ${ARGS[*]}" | tee -a "$QOUT/driver.log"
rc=${PIPESTATUS[0]}
state after
[ "$MODE" = speed ] && quiet off
scp -q "node0:$QRUN0/served.json" "$QOUT/" 2>/dev/null
ql_log "client exit $rc"
cleanup
trap - EXIT
exit "$rc"
