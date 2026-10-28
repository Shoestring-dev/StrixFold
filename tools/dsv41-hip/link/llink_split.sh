#!/bin/bash
# l-link: the served engine (deploy-link) with TF_DS_PREFILL_SPLIT off, then on, then off again; the 2k/4k prompts of
# llink_prompt.py timed inside quiet windows on both boxes; rank 1's prompt-logits SHA-256 kept. Locks: node0's
# (a long wait holds nothing), then node1's within 20 s, else both given back and retried; held for the whole
# session (three server starts, ~6 min); every holder, rank and quiet window released on exit.
set -u
D=${D:-031e229803c6}
MODES=${MODES:-"off on off"}
REPS=${REPS:-2}
T1="~/tfp/deploy-link/$D/tools/dsv41-hip"
T2=$HOME/tfp/deploy-link/$D/tools/dsv41-hip
OUT=$HOME/tfp/link/logs/split-$(date +%m%d-%H%M); mkdir -p "$OUT"
log() { echo "$(date +%T) $*" | tee -a "$OUT/session.log"; }
RUN_ID=; MODE=; QUIET=
r0() { ssh -n node0 "RUN_ID=$RUN_ID TF_SERVE_CONF=\$HOME/tfp/run/conf/llink-$MODE.conf $T1/serve-rank0.sh $*"; }
r1() { RUN_ID=$RUN_ID TF_SERVE_CONF=$HOME/tfp/run/conf/llink-$MODE.conf $T2/serve-rank1.sh "$@"; }
release() {
  ssh -n node0 'p=$(cat /tmp/llink-lock.pid 2>/dev/null); [ -n "$p" ] && kill $p; rm -f /tmp/llink-lock.pid'
  p=$(cat /tmp/llink-lock.pid 2>/dev/null); [ -n "$p" ] && kill $p; rm -f /tmp/llink-lock.pid
}
cleanup() {
  [ -n "$QUIET" ] && { ~/bin/halo-quiet off "$QUIET"; ssh -n node0 "~/bin/halo-quiet off $QUIET"; }
  if [ -n "$RUN_ID" ]; then r0 stop; r1 stop; fi
  release
  log "session end; locks released"
}
trap cleanup EXIT
lock_both() {
  while true; do
    ssh -n node0 "rm -f /tmp/llink-lock.pid; setsid -f flock -w 1800 /var/tmp/halo-gpu.lock sh -c 'echo \$\$ > /tmp/llink-lock.pid; exec sleep 7200' </dev/null >/dev/null 2>&1; for i in \$(seq 1 1810); do [ -s /tmp/llink-lock.pid ] && exit 0; sleep 1; done; exit 1" || return 1
    rm -f /tmp/llink-lock.pid
    setsid -f flock -w 20 /var/tmp/halo-gpu.lock sh -c 'echo $$ > /tmp/llink-lock.pid; exec sleep 7200' </dev/null >/dev/null 2>&1
    for i in $(seq 1 25); do [ -s /tmp/llink-lock.pid ] && return 0; sleep 1; done
    ssh -n node0 'kill $(cat /tmp/llink-lock.pid); rm -f /tmp/llink-lock.pid'
    log "node1 lock busy; gave node0 back, retrying in 30 s"
    sleep 30
  done
}
log "waiting for the locks (node0 first)"
lock_both || { log "node0 lock not acquired"; exit 1; }
log "both locks held; load node0 $(ssh -n node0 cat /proc/loadavg) / node1 $(cut -d' ' -f1-3 /proc/loadavg)"
scp -q ~/tfp/link/llink_prompt.py node0:/tmp/llink_prompt.py
k=0
for MODE in $MODES; do
  k=$((k + 1))
  RUN_ID=llink-$MODE-$k-$(date +%H%M%S)
  log "== $RUN_ID: start rank 1, rank 0"
  r1 start --locked || exit 1
  r0 start --locked || exit 1
  r1 wait-linked || exit 1
  for i in $(seq 1 180); do ssh -n node0 "curl -sf http://127.0.0.1:18080/health" >/dev/null && break; sleep 2; done
  QUIET=llink-$RUN_ID
  ssh -n node0 "~/bin/halo-quiet on $QUIET" >/dev/null; ~/bin/halo-quiet on "$QUIET" >/dev/null; sleep 6
  log "$RUN_ID load: node0 $(ssh -n node0 cat /proc/loadavg) / node1 $(cut -d' ' -f1-3 /proc/loadavg)"
  ssh -n node0 "python3 /tmp/llink_prompt.py $REPS" | tee "$OUT/$RUN_ID.jsonl"
  log "$RUN_ID load after: node0 $(ssh -n node0 cat /proc/loadavg) / node1 $(cut -d' ' -f1-3 /proc/loadavg)"
  ~/bin/halo-quiet off "$QUIET" >/dev/null; ssh -n node0 "~/bin/halo-quiet off $QUIET" >/dev/null; QUIET=
  ssh -n node0 "grep -h prompt_logits ~/tfp/run/serve/$RUN_ID/rank0.log" > "$OUT/$RUN_ID.sha" || true
  grep -h "prefill\|split\|error\|refused\|collectives" "$HOME/tfp/run/serve/$RUN_ID/rank1.log" | head -20 > "$OUT/$RUN_ID.rank1-notes"
  ssh -n node0 "grep -h 'reason=\|error' ~/tfp/run/serve/$RUN_ID/rank0.log | tail -20" > "$OUT/$RUN_ID.rank0-notes"
  r0 stop; for i in $(seq 1 120); do r1 status 2>/dev/null | grep -q "running (" || break; sleep 1; done; r1 stop
  RUN_ID=
done
log "results in $OUT"
