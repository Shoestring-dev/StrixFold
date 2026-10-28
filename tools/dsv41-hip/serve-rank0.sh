#!/bin/bash
# serve-rank0.sh start [--locked] | ready | chat PROMPT | stop | status: rank 0 of the TP2 server (node0),
# tensorfold-native serve, from serve.conf (serve_lib.sh resolves it; RUN_ID names the run, default the UTC time).
#   start   resolve, check the inputs, write RUN/resolved-rank0.json, start the server in the background
#           (RUN/rank0.log, RUN/rank0.pid) under this box's GPU lock (--locked: the caller holds it already)
#   ready   until /health answers (the weights loaded, rank 1 linked, the engine up) within READY_TIMEOUT_S, then the
#           probe chat request (PROBE_PROMPT, greedy): ready only when its reply contains PROBE_EXPECT; the reply and
#           its timing go to RUN/probe.json
#   chat    one chat request, its reply and timing printed
#   bench   the timed requests (BENCH_CASES: the code and prose decode prompts, the BENCH_REPEAT prefill prompt; each
#           BENCH_MAX_TOKENS), streamed: TTFT, prefill and decode tok/s, the server's /metrics after them, uptime;
#           RUN/bench.json (first_light.sh runs it inside quiet windows)
#   stop    SIGTERM (a clean stop: the engine closes and rank 1 hears done), SIGKILL after STOP_TIMEOUT_S
#   check   resolve and check only: RUN/resolved-rank0.json printed (identities: deploy commit, kit, config)
#   lock [WAIT_S] / unlock   hold (waiting at most WAIT_S, default LOCK_WAIT_S) / release this box's GPU lock for
#           the run (first_light.sh takes node0's, then node1's)
#   status  running or not, /health, the log's last line
set -euo pipefail
source "$(dirname "$0")/serve_lib.sh"
serve_load 0
RUN_ID=${RUN_ID:-$(serve_latest)}
RUN=$RUN_ROOT/${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
URL=http://$SERVE_HOST:$SERVE_PORT

# chat PROMPT MAX_TOKENS OUT: one greedy chat request; OUT gets the reply, the usage and the wall time
chat() {
  python3 - "$URL" "$SERVE_NAME" "$1" "$2" "$PROBE_TIMEOUT_S" "$3" <<'PY'
import json, sys, time, urllib.request
url, name, prompt, max_tokens, timeout, out = sys.argv[1:7]
body = json.dumps({"model": name, "messages": [{"role": "user", "content": prompt}], "max_tokens": int(max_tokens),
                   "temperature": 0}).encode()
t0 = time.monotonic()
req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=float(timeout)) as r:
    reply = json.load(r)
wall = time.monotonic() - t0
text = reply["choices"][0]["message"].get("content") or ""
usage = reply.get("usage", {})
done = usage.get("completion_tokens") or 0
rec = {"prompt": prompt, "reply": text, "usage": usage, "wall_s": round(wall, 3),
       "completion_tok_per_s_wall": round(done / wall, 2) if wall > 0 and done else None, "response": reply}
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("reply", "usage", "wall_s", "completion_tok_per_s_wall")}))
PY
}

# bench OUT: the timed requests (BENCH_CASES), streamed and greedy, one after another (bench_stream.py)
bench() {
  python3 "$(dirname "$0")/bench_stream.py" "$URL" "$SERVE_NAME" "${BENCH_CASES:-code,prose,prefill}" "$BENCH_REPEAT" \
    "$BENCH_MAX_TOKENS" "$PROBE_TIMEOUT_S" "$1"
}

case ${1:-} in
  start)
    mkdir -p "$RUN"; ln -sfn "$(basename "$RUN")" "$RUN_ROOT/latest"
    [ -z "$(serve_pid "$RUN/rank0.pid")" ] || die "rank 0 already runs ($(cat "$RUN/rank0.pid"))"
    serve_check; serve_idle; serve_env; serve_cmd; serve_resolved "$RUN/resolved-rank0.json"
    serve_spawn "$RUN/rank0.pid" "$RUN/rank0.log" "$([ "${2:-}" = --locked ] && echo 1 || echo 0)"
    echo "rank 0 started ($(cat "$RUN/rank0.pid")): $RUN";;
  ready)
    for _ in $(seq 1 "$READY_TIMEOUT_S"); do
      curl -fsS -m 5 "$URL/health" >/dev/null 2>&1 && break
      [ -n "$(serve_pid "$RUN/rank0.pid")" ] || { echo "rank 0 exited:"; tail -5 "$RUN/rank0.log"; exit 1; }
      sleep 1
    done
    curl -fsS -m 5 "$URL/health" >/dev/null || die "no /health within ${READY_TIMEOUT_S}s"
    chat "$PROBE_PROMPT" "$PROBE_MAX_TOKENS" "$RUN/probe.json"
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if sys.argv[2].lower() in r["reply"].lower() else 1)' \
      "$RUN/probe.json" "$PROBE_EXPECT" || die "the probe's reply does not contain '$PROBE_EXPECT' (RUN/probe.json)"
    echo "ready: $URL ($RUN)";;
  chat) mkdir -p "$RUN"; chat "${2:?PROMPT}" "${3:-$MAX_TOKENS}" "$RUN/chat-$(date -u +%H%M%S).json";;
  bench) mkdir -p "$RUN"; bench "$RUN/bench.json";;
  check) mkdir -p "$RUN"; serve_check; serve_env; serve_cmd; serve_resolved "$RUN/resolved-rank0.json"; cat "$RUN/resolved-rank0.json";;
  lock) mkdir -p "$RUN"; serve_lock_hold "$RUN" "${2:-}"; echo "GPU lock held on $(hostname) ($(cat "$RUN/lock.pid"))";;
  unlock) serve_lock_release "$RUN"; echo "GPU lock released on $(hostname)";;
  stop) serve_stop "$RUN/rank0.pid" "rank 0";;
  status)
    pid=$(serve_pid "$RUN/rank0.pid"); echo "rank 0 ${pid:+running ($pid)}${pid:-not running}: $RUN"
    curl -fsS -m 5 "$URL/health" 2>/dev/null && echo || echo "no /health"
    tail -1 "$RUN/rank0.log" 2>/dev/null || true;;
  *) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2;;
esac
