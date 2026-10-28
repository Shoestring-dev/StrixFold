#!/bin/bash
# gpu_tests.sh DEPLOY_DIR OUT_DIR: the dsv41 HIP GPU test programs of a deployment (DEPLOY_DIR/native-hip/test: smoke,
# ABI, glue) on this box under its GPU lock, as run_gpu_tests.sh runs them from a build tree (same arguments, same
# per-program receipts), with the deployment's identity in OUT_DIR/run.json. Exit 0 only if every program passes.
set -u
dep=${1:?DEPLOY_DIR}; out=${2:?OUT_DIR}
bin=$dep/native-hip/test
mkdir -p "$out"
programs=(tf-dsv41-hip-smoke tf-dsv41-hip-abi-test tf-dsv41-hip-glue-test)
commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$dep/DEPLOY.json")
exec 9>/var/tmp/halo-gpu.lock
flock -w 1800 9 || { echo "GPU lock not acquired in 30 min" >&2; exit 3; }
scratch=$(mktemp -d /tmp/q-qual-gputest.XXXXXX)
status=0
rows=()
for p in "${programs[@]}"; do
  extra=()
  [ "$p" = tf-dsv41-hip-abi-test ] && extra=(--kit "$bin/kit" --scratch "$scratch")
  [ "$p" = tf-dsv41-hip-glue-test ] && extra=(--kit "$bin/dsv41-kit")
  if [ ! -x "$bin/$p" ]; then rows+=("{\"program\": \"$p\", \"exit\": 127, \"seconds\": 0, \"receipt\": null}"); status=1; continue; fi
  start=$(date +%s.%N)
  timeout 600 "$bin/$p" "${extra[@]}" --receipt "$out/$p.json" > "$out/$p.log" 2>&1
  rc=$?
  end=$(date +%s.%N)
  [ $rc -ne 0 ] && status=1
  rows+=("$(printf '{"program": "%s", "exit": %d, "seconds": %.1f, "receipt": "%s.json"}' "$p" $rc "$(echo "$end - $start" | bc)" "$p")")
done
flock -u 9
exec 9>&-
rm -rf "$scratch"
{
  echo "{"
  echo "  \"host\": \"$(hostname)\", \"utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"deploy\": \"$dep\", \"deploy_commit\": \"$commit\","
  echo "  \"hipcc\": \"$(hipcc --version 2>/dev/null | head -1)\", \"kernel\": \"$(uname -r)\","
  echo "  \"test_kit_manifest_sha256\": \"$(sha256sum "$bin/kit/manifest.json" | cut -d' ' -f1)\","
  echo "  \"dsv41_kit_manifest_sha256\": \"$(sha256sum "$bin/dsv41-kit/manifest.json" | cut -d' ' -f1)\","
  echo "  \"programs\": ["
  (IFS=,; echo "    ${rows[*]}")
  echo "  ],"
  echo "  \"all_passed\": $([ $status -eq 0 ] && echo true || echo false)"
  echo "}"
} > "$out/run.json"
cat "$out/run.json"
exit $status
