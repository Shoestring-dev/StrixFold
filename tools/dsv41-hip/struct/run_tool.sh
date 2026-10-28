#!/bin/bash
# run_tool.sh ARGS...: builds grammar_tool.zig against the grammar module (ReleaseFast, host only) and runs it with
# ARGS, both under node1's house rules for background CPU work (upper cores, capped, low priority; a quiet window may
# pause it).
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
zig=${ZIG:-$HOME/opt/zig/zig}
out=${TOOL_OUT:-/tmp/sstruct/grammar_tool}
mkdir -p "$(dirname "$out")"
systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=16G taskset -c 8-15,24-31 nice -n 19 ionice -c3 \
  env ZIG_LOCAL_CACHE_DIR="$root/.zig-cache" ZIG_GLOBAL_CACHE_DIR="$(dirname "$root")/.zig-global-$(basename "$root")" \
  "$zig" build-exe -O ReleaseFast -lc --dep grammar -Mroot="$root/tools/dsv41-hip/struct/grammar_tool.zig" \
  -Mgrammar="$root/zig/src/core/lanes/grammar/grammar.zig" -femit-bin="$out"
exec systemd-run --user --scope --quiet -p CPUQuota=100% -p MemoryMax=4G taskset -c 8-15,24-31 nice -n 19 ionice -c3 "$out" "$@"
