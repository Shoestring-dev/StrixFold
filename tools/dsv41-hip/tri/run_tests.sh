#!/bin/bash
# The w-tri correctness suite on the AOT kit (one GPU-lock acquisition). Usage: run_tests.sh [test files...]
set -u
cd "$(dirname "$0")"
export TRITON_CACHE_DIR=$HOME/tfp/tri/.triton-cache
PY=$HOME/venvs/triton/bin/python
tests=("$@")
[ ${#tests[@]} -eq 0 ] && tests=(t_cover.py t_basic.py t_hc.py t_route.py t_index.py t_attn.py t_markov.py t_gemm.py t_native.py t_real.py t_hipattn.py t_hipindex.py t_hiphc.py t_hipgate.py t_hiprope.py t_wmma.py)
stamp=$(date +%Y%m%dT%H%M%S)
logs=${TRI_WORK:-$HOME/tfp/tri}/logs
mkdir -p "$logs"
rc=0
for t in "${tests[@]}"; do
  echo "=== $t $(date +%T)"
  timeout 900 $PY "$t" > "$logs/${t%.py}-$stamp.log" 2>&1
  r=$?
  echo "    exit $r, $(grep -c '^\[PASS\]' $logs/${t%.py}-$stamp.log) pass, $(grep -c '^\[FAIL\]' $logs/${t%.py}-$stamp.log) fail"
  [ $r -ne 0 ] && rc=1 && grep -E '^\[FAIL\]|Error|Traceback' "$logs/${t%.py}-$stamp.log" | head -8
done
exit $rc
