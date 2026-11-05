#!/bin/bash
# verify_inputs.sh BUNDLE RANK MODEL_DIR ENGRAM_DIR MANIFEST_SHA256 [POOL] [-- extra verify_inputs.py flags]: one box's
# inputs for one TP2 rank, checked before that rank loads them (BUILD_AND_RUN.md section 3): the bundle manifest must hash to
# MANIFEST_SHA256 (the value serve.conf pins as BUNDLE_SHA256); the bundle's files re-hashed against it (the other rank's
# cache may be absent); the chat template, token map, Engram constants and RoPE tables against identities kept in this
# repository (assets/); the checkpoint and both Engram shards authenticated against the pinned HF metadata; and this
# rank's cache byte-checked in full (tf-dsv41-check --bytes 1, exact entry and byte counts). Report:
# BUNDLE.verify-rank<R>-<host>.json. Exit 0 pass, 3 pass-with-skips (a --skip-* flag), 1 fail.
# Same environment and resource rules as prepare_inputs.sh (TFP_INPUT_TOOLS, TFP_INPUT_PYTHON, TFP_BIN; the input tools
# are the ones vendored here (see kit_pin.sh); house CPU/memory confinement; no lock: the quiet lock is retired; a
# halo-quiet window may pause it). About 520-730 GB of reads per box (about 15 min).
set -euo pipefail
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2; }
[ $# -ge 5 ] || usage
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
if [ -z "${TFP_SCOPED:-}" ]; then
  export TFP_SCOPED=1
  exec systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryHigh=4G -p MemoryMax=8G -- \
    taskset -c 8-15,24-31 /usr/bin/nice -n 19 /usr/bin/ionice -c3 "$0" "$@"
fi
BUNDLE=$1 RANK=$2 MODEL=$3 ENGRAM=$4 MSHA=$5; shift 5
POOL=8192; if [ $# -gt 0 ] && [ "$1" != "--" ]; then POOL=$1; shift; fi
[ "${1:-}" = "--" ] && shift
. "$HERE/kit_pin.sh"
TOOLS=$(input_tools "$HERE")
PY=${TFP_INPUT_PYTHON:-$HOME/venvs/tfp-inputs/bin/python}
BIN=${TFP_BIN:-$ROOT/zig-out/bin}
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
exec "$PY" "$HERE/verify_inputs.py" "$BUNDLE" --rank "$RANK" --expect-manifest-sha256 "$MSHA" --model "$MODEL" \
  --engram-dir "$ENGRAM" --check-bin "$BIN/tf-dsv41-check" --tools "$TOOLS" --kit-commit "$TFP_KIT_COMMIT" --pool "$POOL" \
  --output "${BUNDLE%/}.verify-rank$RANK-$(hostname).json" "$@"
