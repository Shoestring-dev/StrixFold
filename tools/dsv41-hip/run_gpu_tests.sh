#!/bin/bash
# Run the dsv41 HIP GPU test programs under the shared GPU lock and write one receipt directory.
#   [EXTRA_KIT=DIR] [CROSS="LABEL=DIR ..."] tools/dsv41-hip/run_gpu_tests.sh [OUT_DIR]   (default: ~/tfp/arch/receipts/<UTC time>)
# EXTRA_KIT: another kit for the ABI test to open through the loader (e.g. the imported Triton kit).
# CROSS: other implementations of the glue kernels for the glue test's byte-for-byte cross-check (cross_objects.sh).
# Build first: zig build dsv41-hip-gpu-test (programs land in zig-out/native-hip/test). Each program is bounded by a
# timeout; the receipt records the git state, the toolchain and each program's own JSON. Exit 0 only if all pass.
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:-$HOME/tfp/arch/receipts/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$out"
bin=$root/zig-out/native-hip/test
programs=(tf-dsv41-hip-smoke)
[ -x "$bin/tf-dsv41-hip-abi-test" ] && programs+=(tf-dsv41-hip-abi-test)
[ -x "$bin/tf-dsv41-hip-glue-test" ] && programs+=(tf-dsv41-hip-glue-test)

{
  echo "{"
  echo "  \"host\": \"$(hostname)\","
  echo "  \"utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"git_head\": \"$(git -C "$root" rev-parse HEAD)\","
  echo "  \"git_tree\": \"$(git -C "$root" rev-parse 'HEAD^{tree}')\","
  echo "  \"git_dirty\": $( [ -z "$(git -C "$root" status --porcelain --untracked-files=no)" ] && echo false || echo true ),"
  echo "  \"hipcc\": \"$(hipcc --version 2>/dev/null | head -1)\","
  echo "  \"kernel\": \"$(uname -r)\","
  [ -f "$bin/kit/manifest.json" ] && echo "  \"test_kit_manifest_sha256\": \"$(sha256sum "$bin/kit/manifest.json" | cut -d' ' -f1)\","
  [ -f "$bin/dsv41-kit/manifest.json" ] && echo "  \"dsv41_kit_manifest_sha256\": \"$(sha256sum "$bin/dsv41-kit/manifest.json" | cut -d' ' -f1)\","
  echo "  \"programs\": ["
} > "$out/run.json"

status=0
first=1
exec 9>/var/tmp/halo-gpu.lock
if ! flock -w 1800 9; then echo "GPU lock not acquired in 30 min" >&2; exit 3; fi
scratch=$(mktemp -d /tmp/dsv41-hip-test.XXXXXX)
for p in "${programs[@]}"; do
  extra=()
  [ "$p" = tf-dsv41-hip-abi-test ] && extra=(--kit "$bin/kit" --scratch "$scratch")
  [ "$p" = tf-dsv41-hip-abi-test ] && [ -n "${EXTRA_KIT:-}" ] && extra+=(--extra-kit "$EXTRA_KIT")
  [ "$p" = tf-dsv41-hip-glue-test ] && extra=(--kit "$bin/dsv41-kit")
  if [ "$p" = tf-dsv41-hip-glue-test ] && [ -n "${CROSS:-}" ]; then for c in $CROSS; do extra+=(--cross "$c"); done; fi
  start=$(date +%s.%N)
  timeout 600 "$bin/$p" "${extra[@]}" --receipt "$out/$p.json" > "$out/$p.log" 2>&1
  rc=$?
  end=$(date +%s.%N)
  [ $rc -ne 0 ] && status=1
  [ $first -eq 1 ] || echo "," >> "$out/run.json"
  first=0
  printf '    {"program": "%s", "exit": %d, "seconds": %.1f, "receipt": "%s.json"}' "$p" $rc "$(echo "$end - $start" | bc)" "$p" >> "$out/run.json"
  echo "$p exit $rc"
done
flock -u 9
rm -rf "$scratch"
printf '\n  ],\n  "all_passed": %s\n}\n' "$( [ $status -eq 0 ] && echo true || echo false )" >> "$out/run.json"
echo "receipt: $out"
exit $status
