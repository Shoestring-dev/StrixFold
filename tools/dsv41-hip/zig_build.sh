#!/bin/bash
# zig build for the dsv41 HIP port under node1's house rules (~/tfp/PLAN.md): cores 8-15 and 24-31, 800% CPU, memory
# high 12 GiB / max 16 GiB, nice 19 and idle I/O, -j8 (a quiet window may pause it: no deadlines); HIPCC_COMPILE_FLAGS_APPEND / HIPCC_LINK_FLAGS_APPEND removed (explicit device
# flags are part of the build contract); this worktree's own Zig caches.
#   tools/dsv41-hip/zig_build.sh STEP... [-Doption=...]
# The user systemd manager here does not delegate the cpuset controller, so AllowedCPUs alone is not enforced: taskset
# pins the cores, the scope enforces the CPU quota and memory limits.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
zig=${ZIG:-$HOME/opt/zig/zig}
cd "$root"
exec systemd-run --user --scope --quiet -p AllowedCPUs=8-15,24-31 -p CPUQuota=800% -p MemoryHigh=12G -p MemoryMax=16G \
    env -u HIPCC_COMPILE_FLAGS_APPEND -u HIPCC_LINK_FLAGS_APPEND \
    ZIG_LOCAL_CACHE_DIR="$root/.zig-cache" ZIG_GLOBAL_CACHE_DIR="$(dirname "$root")/.zig-global-$(basename "$root")" \
    taskset -c 8-15,24-31 nice -n 19 ionice -c3 "$zig" build -j8 "$@"
