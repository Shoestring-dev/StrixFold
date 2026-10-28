# qual_lib.sh: sourced by the served qualification drivers (qual_serve.sh, served_faults.sh). Runs on node1 (rank 1's
# box, the build box); reaches node0 (rank 0) by ssh. Both ranks start through the deployment's own serve-rank0.sh /
# serve-rank1.sh with one configuration file, which must exist at the same path under each box's $HOME.
#   QCONF   the configuration, '$HOME/...' (single-quoted: each box expands its own $HOME)
# The GPU lock holders are this file's own (setsid flock ... sleep QLOCK_MAX_S), their pids in
# ~/tfp/qual/locks/<QRUN>.pid on each box (one file per run: two drivers never touch each other's holder); every driver
# releases them on every exit path, and a holder dies by itself after QLOCK_MAX_S; the ranks start with --locked.
#   QRUN    the run id (rank directories RUN_ROOT/QRUN on both boxes, from the configuration's RUN_ROOT)
#   QCONF_RANK0 / QCONF_RANK1  (optional) that rank from another configuration: the peer-mismatch cases
# The lock rules: node0's GPU lock first, then node1's within LOCK2_WAIT_S or neither; held for one run,
# released on every exit path; servers stopped by pid; nothing left behind.
# shellcheck shell=bash

# ql_quiet on|off TAG: the optional quiet-window hook on both boxes (QUIET_BIN, default ~/bin/halo-quiet, called as
# QUIET_BIN on|off TAG; a box without it is skipped): node0 first on, both off. It holds other work on a box while a
# timed run goes; the measurements in docs/RESULTS.md were taken inside such windows.
ql_quiet() {
  local q='b=${QUIET_BIN:-$HOME/bin/halo-quiet}; [ -x "$b" ] && "$b"'
  if [ "$1" = on ]; then ssh -n node0 "$q on $2 || true"; eval "$q on \"\$2\" || true"
  else eval "$q off \"\$2\" || true"; ssh -n node0 "$q off $2 || true"; fi
}

QTOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCK1_WAIT_S=${LOCK1_WAIT_S:-1800}
LOCK2_WAIT_S=${LOCK2_WAIT_S:-60}
QLOCK_MAX_S=${QLOCK_MAX_S:-1800}   # a lock holder ends by itself after this (the run's own length; never longer)
QHOME0=$(ssh -n node0 'echo $HOME')

ql_log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$QOUT/driver.log" >&2; }

# ql_init: QCONF and QRUN set; the configuration resolved on this box for the deployment and run root.
ql_init() {
  : "${QCONF:?}" "${QRUN:?}"
  QCONF1=${QCONF//\$HOME/$HOME}
  QCONF0=${QCONF//\$HOME/$QHOME0}
  [ -f "$QCONF1" ] || { echo "no $QCONF1" >&2; exit 2; }
  ssh -n node0 "test -f $QCONF0" || { echo "no $QCONF0 on node0" >&2; exit 2; }
  cmp -s "$QCONF1" <(ssh -n node0 "cat $QCONF0") || { echo "the configuration differs between the boxes" >&2; exit 2; }
  # shellcheck disable=SC1090
  (source "$QCONF1"; echo "$DEPLOY_DIR" "$RUN_ROOT" "$SERVE_PORT" "$CONTEXT" "$DRAFTS" "$GRAPHS" "$SERVE_NAME") > /tmp/ql-$$.txt
  read -r QDEPLOY QROOT QPORT QCTX QDRAFTS QGRAPHS QNAME < /tmp/ql-$$.txt; rm -f /tmp/ql-$$.txt
  QT1=${QDEPLOY//\$HOME/$HOME}/tools/dsv41-hip
  QT0=${QDEPLOY//\$HOME/$QHOME0}/tools/dsv41-hip
  QRUN1=${QROOT//\$HOME/$HOME}/$QRUN
  QRUN0=${QROOT//\$HOME/$QHOME0}/$QRUN
  QOUT=${QOUT_ROOT:-$HOME/tfp/qual/out}/$QRUN
  QLOCK1=$HOME/tfp/qual/locks/$QRUN.pid; QLOCK0="\$HOME/tfp/qual/locks/$QRUN.pid"   # the lock holders' pids, per run
  mkdir -p "$HOME/tfp/qual/locks"; ssh -n node0 "mkdir -p ~/tfp/qual/locks"
  # a mismatch case: one rank from another configuration (the same file on both boxes)
  [ -n "${QCONF_RANK0:-}" ] && QCONF0=${QCONF_RANK0//\$HOME/$QHOME0}
  [ -n "${QCONF_RANK1:-}" ] && QCONF1=${QCONF_RANK1//\$HOME/$HOME}
  mkdir -p "$QOUT" "$QRUN1"
  ssh -n node0 "mkdir -p $QRUN0 ~/tfp/qual/bin"
  scp -q "$QTOOLS/serve_qual.py" "$QTOOLS/memledger.py" "$QTOOLS/fault_client.py" node0:tfp/qual/bin/
  cp "$QTOOLS/memledger.py" "$HOME/tfp/qual/bin/" 2>/dev/null || { mkdir -p "$HOME/tfp/qual/bin"; cp "$QTOOLS/memledger.py" "$HOME/tfp/qual/bin/"; }
}

r0() { ssh -n node0 "RUN_ID=$QRUN TF_SERVE_CONF=$QCONF0 $QT0/serve-rank0.sh $*"; }
r1() { RUN_ID=$QRUN TF_SERVE_CONF=$QCONF1 "$QT1/serve-rank1.sh" "$@"; }

# ql_lock: both GPU locks, node0 first; neither if node1's is not free within LOCK2_WAIT_S, then another attempt after
# LOCK_RETRY_S (LOCK_TRIES attempts in all; nothing is held between attempts). Holders run in their own session
# (setsid), their pids in ~/tfp/qual/locks/<QRUN>.pid on each box.
ql_lock() {
  local try
  for try in $(seq 1 "${LOCK_TRIES:-8}"); do
    ql_lock_once && return 0
    [ "$try" -lt "${LOCK_TRIES:-8}" ] && { ql_log "lock attempt $try failed; again in ${LOCK_RETRY_S:-45}s"; sleep "${LOCK_RETRY_S:-45}"; }
  done
  return 1
}

ql_lock_once() {
  ql_log "lock node0 (wait <= ${LOCK1_WAIT_S}s)"
  ssh -n node0 "rm -f $QLOCK0; setsid -f flock -w $LOCK1_WAIT_S /var/tmp/halo-gpu.lock sh -c 'echo \$\$ > $QLOCK0; exec sleep $QLOCK_MAX_S' </dev/null >/dev/null 2>&1;
    for i in \$(seq 1 $LOCK1_WAIT_S); do [ -s $QLOCK0 ] && exit 0; sleep 1; done; exit 1" || { ql_log "node0 lock not free"; sleep 2; ql_unlock; return 1; }
  ql_log "lock node1 (wait <= ${LOCK2_WAIT_S}s)"
  rm -f "$QLOCK1"
  setsid -f flock -w "$LOCK2_WAIT_S" /var/tmp/halo-gpu.lock sh -c "echo \$\$ > $QLOCK1; exec sleep $QLOCK_MAX_S" </dev/null >/dev/null 2>&1
  local i
  for i in $(seq 1 "$LOCK2_WAIT_S"); do [ -s "$QLOCK1" ] && { ql_log "both locks held"; return 0; }; sleep 1; done
  ql_log "node1 lock not free within ${LOCK2_WAIT_S}s: releasing node0's"
  sleep 2   # a holder that got the lock as the wait ended is released too
  ql_unlock
  return 1
}

ql_unlock() {
  [ -s "$QLOCK1" ] && kill "$(cat "$QLOCK1")" 2>/dev/null; rm -f "$QLOCK1"
  ssh -n node0 "[ -s $QLOCK0 ] && kill \$(cat $QLOCK0) 2>/dev/null; rm -f $QLOCK0" || true
  ql_log "locks released"
}

# ql_phase NAME: the phase label both samplers record
ql_phase() { echo "$1" > "$QRUN1/qphase"; ssh -n node0 "echo $1 > $QRUN0/qphase"; ql_log "phase $1"; }

ql_sampler_start() {
  rm -f "$QRUN1/qstop"; ssh -n node0 "rm -f $QRUN0/qstop"
  ql_phase start
  setsid -f nohup python3 "$HOME/tfp/qual/bin/memledger.py" sample "$QOUT/mem-node1.jsonl" "$QRUN1/qphase" "$QRUN1/qstop" "$QRUN1/rank1.pid" </dev/null >/dev/null 2>&1
  ssh -n node0 "setsid -f nohup python3 ~/tfp/qual/bin/memledger.py sample $QRUN0/mem-node0.jsonl $QRUN0/qphase $QRUN0/qstop $QRUN0/rank0.pid </dev/null >/dev/null 2>&1"
}

ql_sampler_stop() {
  touch "$QRUN1/qstop"; ssh -n node0 "touch $QRUN0/qstop"
  sleep 2
  scp -q "node0:$QRUN0/mem-node0.jsonl" "$QOUT/" 2>/dev/null || true
}

# ql_start: rank 1, then rank 0, linked and /health answering (READY within the configuration's timeout)
ql_start() {
  ql_phase load
  r1 start --locked >> "$QOUT/driver.log" 2>&1 || return 1
  r0 start --locked >> "$QOUT/driver.log" 2>&1 || return 1
  r1 wait-linked >> "$QOUT/driver.log" 2>&1 || return 1
  local i
  for i in $(seq 1 600); do
    ssh -n node0 "curl -fsS -m 5 http://127.0.0.1:$QPORT/health" >/dev/null 2>&1 && { ql_log "ready"; return 0; }
    [ -n "$(ql_pid0)" ] || { ql_log "rank 0 exited before /health"; return 1; }
    sleep 1
  done
  return 1
}

# ql_env_refusal: true when rank 0 refused to load because node0's memory was not free (U25: other work resident);
# drivers record such a start as unrun (the box was not idle), never as a result of the port
ql_env_refusal() { ssh -n node0 "grep -q 'do not fit the' $QRUN0/rank0.log 2>/dev/null"; }

ql_pid0() { ssh -n node0 "f=$QRUN0/rank0.pid; [ -f \$f ] && kill -0 \$(cat \$f) 2>/dev/null && cat \$f" 2>/dev/null || true; }
ql_pid1() { [ -f "$QRUN1/rank1.pid" ] && kill -0 "$(cat "$QRUN1/rank1.pid")" 2>/dev/null && cat "$QRUN1/rank1.pid"; return 0; }

ql_stop() {
  ql_phase stop
  r0 stop >> "$QOUT/driver.log" 2>&1 || true
  local i
  for i in $(seq 1 60); do [ -z "$(ql_pid1)" ] && break; sleep 1; done
  r1 stop >> "$QOUT/driver.log" 2>&1 || true
}

# ql_maps OUT: the shared objects both rank processes have mapped; fails if any NVIDIA/CUDA library is among them
ql_maps() {
  local p0 p1
  p0=$(ql_pid0); p1=$(ql_pid1)
  { echo "== rank0 $p0"; ssh -n node0 "awk '{print \$6}' /proc/$p0/maps | grep '\.so' | sort -u";
    echo "== rank1 $p1"; awk '{print $6}' "/proc/$p1/maps" | grep '\.so' | sort -u; } > "$1"
  if grep -i -E 'libcuda|libnvidia|nccl|cublas|cudart|nvrtc|libnvJit' "$1"; then return 1; fi
  grep -q libamdhip64 "$1"
}

# ql_copy_logs: both ranks' logs and resolved configurations into QOUT
ql_copy_logs() {
  scp -q "node0:$QRUN0/rank0.log" "node0:$QRUN0/resolved-rank0.json" "$QOUT/" 2>/dev/null || true
  cp "$QRUN1/rank1.log" "$QRUN1/resolved-rank1.json" "$QOUT/" 2>/dev/null || true
}
