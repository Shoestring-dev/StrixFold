#!/bin/bash
# lane_rank.sh CONF RANK DIR [ARG...]: one rank of the lane gate (tf-dsv41-lanes) of configuration CONF on this box,
# from this script's own deployment: serve_lib.sh resolves CONF for RANK exactly as serving does (the whole environment;
# the command rank 1 serves with, for rank 0 with its own rank and cache), ARGs appended (rank 0: --requests FILE ...).
# DIR gets rankR.cmd (argv and environment), resolved-rankR.json, rankR.pid (the gate's own pid, written before exec),
# rankR.jsonl (stdout) and rankR.err (stderr: the --logits-sha lines); on ENGINE_CPUS when the configuration sets them,
# as serve_spawn. Runs in the foreground (callers detach it); the caller holds this box's GPU lock (qual/lane_gate.sh).
set -euo pipefail
export TF_SERVE_CONF=${1:?CONF}
R=${2:?RANK}; DIR=${3:?DIR}; shift 3
source "$(dirname "$0")/../serve_lib.sh"
serve_load "$R"
serve_check; serve_env; serve_lanes_cmd
SERVE_CMD+=("$@")
mkdir -p "$DIR"
serve_resolved "$DIR/resolved-rank$R.json"
{ printf '%q ' "${SERVE_CMD[@]}"; echo; printf '%s\n' "${SERVE_ENV[@]}"; } > "$DIR/rank$R.cmd"
pin=()
[ -n "${ENGINE_CPUS:-}" ] && pin=(taskset -c "$ENGINE_CPUS")
cd "$DIR"
echo $$ > "rank$R.pid"
exec "${pin[@]}" env -i "${SERVE_ENV[@]}" "${SERVE_CMD[@]}" > "rank$R.jsonl" 2> "rank$R.err"
