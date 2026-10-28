#!/bin/bash
# l-link: split numerics and speed on the served engine (deploy-link D). STEPS: conf names, each one server start:
# tfoff / tfon (TF_DS_WINDOW_LOGITS dumps to node0 ~/tfp/link/tf-off|tf-on, llink_tf.py's 18 prompts) and off / on
# (timing: llink_prompt8.py, 4K and 8K twice). Locks node0 then node1 (20 s, else both back), quiet windows around each
# client run, every holder / rank / window released on exit. The lock holders are per session (their pids in
# ~/tfp/link/locks/<session>.pid on each box) and end by themselves after HOLD_S (default 1500 s, the session's bound).
set -u
D=${D:?deploy}
STEPS=${STEPS:-"tfoff tfon off on"}
T1="~/tfp/deploy-link/$D/tools/dsv41-hip"
T2=$HOME/tfp/deploy-link/$D/tools/dsv41-hip
SESS=tf-$(date +%m%d-%H%M%S)
OUT=$HOME/tfp/link/logs/$SESS; mkdir -p "$OUT"
HOLD_S=${HOLD_S:-1500}
LK=tfp/link/locks/$SESS.pid   # under $HOME on each box
mkdir -p ~/tfp/link/locks; ssh -n node0 "mkdir -p ~/tfp/link/locks"
log() { echo "$(date +%T) $*" | tee -a "$OUT/session.log"; }
RUN_ID=; MODE=; QUIET=
r0() { ssh -n node0 "RUN_ID=$RUN_ID TF_SERVE_CONF=\$HOME/tfp/run/conf/llink-$MODE.conf $T1/serve-rank0.sh $*"; }
r1() { RUN_ID=$RUN_ID TF_SERVE_CONF=$HOME/tfp/run/conf/llink-$MODE.conf $T2/serve-rank1.sh "$@"; }
release() {
  ssh -n node0 "p=\$(cat ~/$LK 2>/dev/null); [ -n \"\$p\" ] && kill \$p; rm -f ~/$LK"
  p=$(cat ~/$LK 2>/dev/null); [ -n "$p" ] && kill $p; rm -f ~/$LK
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
    ssh -n node0 "rm -f ~/$LK; setsid -f flock -w 1800 /var/tmp/halo-gpu.lock sh -c 'echo \$\$ > ~/$LK; exec sleep $HOLD_S' </dev/null >/dev/null 2>&1; for i in \$(seq 1 1810); do [ -s ~/$LK ] && exit 0; sleep 1; done; exit 1" || return 1
    rm -f ~/$LK
    setsid -f flock -w 20 /var/tmp/halo-gpu.lock sh -c "echo \$\$ > ~/$LK; exec sleep $HOLD_S" </dev/null >/dev/null 2>&1
    for i in $(seq 1 25); do [ -s ~/$LK ] && return 0; sleep 1; done
    ssh -n node0 "kill \$(cat ~/$LK); rm -f ~/$LK"
    log "node1 lock busy; gave node0 back, retrying in 30 s"
    sleep 30
  done
}
log "waiting for the locks (node0 first)"
lock_both || { log "node0 lock not acquired"; exit 1; }
log "both locks held; load node0 $(ssh -n node0 cat /proc/loadavg) / node1 $(cat /proc/loadavg)"
scp -q ~/tfp/link/llink_tf.py ~/tfp/link/llink_prompt8.py node0:/tmp/
ssh -n node0 "rm -rf ~/tfp/link/tf-off ~/tfp/link/tf-on; mkdir -p ~/tfp/link/tf-off ~/tfp/link/tf-on"
k=0
for MODE in $STEPS; do
  k=$((k + 1))
  RUN_ID=llink-$MODE-$k-$(date +%H%M%S)
  log "== $RUN_ID: start rank 1, rank 0"
  r1 start --locked || exit 1
  r0 start --locked || exit 1
  r1 wait-linked || exit 1
  for i in $(seq 1 180); do ssh -n node0 "curl -sf http://127.0.0.1:18080/health" >/dev/null && break; sleep 2; done
  QUIET=llink-$RUN_ID
  ssh -n node0 "~/bin/halo-quiet on $QUIET" >/dev/null; ~/bin/halo-quiet on "$QUIET" >/dev/null; sleep 6
  log "$RUN_ID load: node0 $(ssh -n node0 cat /proc/loadavg) / node1 $(cat /proc/loadavg)"
  case $MODE in
    tf*) ssh -n node0 "python3 /tmp/llink_tf.py ${TFSET:-a}" | tee "$OUT/$RUN_ID.jsonl";;
    *) ssh -n node0 "python3 /tmp/llink_prompt8.py 2" | tee "$OUT/$RUN_ID.jsonl";;
  esac
  log "$RUN_ID load after: node0 $(ssh -n node0 cat /proc/loadavg) / node1 $(cat /proc/loadavg)"
  ~/bin/halo-quiet off "$QUIET" >/dev/null; ssh -n node0 "~/bin/halo-quiet off $QUIET" >/dev/null; QUIET=
  ssh -n node0 "grep -h prompt_logits ~/tfp/run/serve/$RUN_ID/rank0.log" > "$OUT/$RUN_ID.sha" || true
  ssh -n node0 "grep -h 'reason=error\|error:' ~/tfp/run/serve/$RUN_ID/rank0.log | tail -20" > "$OUT/$RUN_ID.rank0-errors"
  grep -h "error" "$HOME/tfp/run/serve/$RUN_ID/rank1.log" | tail -20 > "$OUT/$RUN_ID.rank1-errors"
  r0 stop; for i in $(seq 1 120); do r1 status 2>/dev/null | grep -q "running (" || break; sleep 1; done; r1 stop
  RUN_ID=
done
log "results in $OUT"
