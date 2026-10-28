#!/bin/bash
# gpu.sh CMD...: one GPU run (PLAN.md): the box's GPU lock for this run only (blocking, 30 min), then a quiet window
# (~/bin/halo-quiet on/off kdec-PID) around it; sclk / load are logged by the harness and here.
exec flock -E 75 -w ${GPU_WAIT:-1800} /var/tmp/halo-gpu.lock bash -c '
tag=kdec-$$
~/bin/halo-quiet on "$tag" >/dev/null
trap "~/bin/halo-quiet off $tag >/dev/null" EXIT
for i in $(seq ${QUIET_WAIT:-60}); do awk "{exit !(\$1 < 2)}" /proc/loadavg && break; sleep 1; done
echo "$(date -Is) gpu.sh $(hostname) load $(cut -d" " -f1-3 /proc/loadavg) after ${i}s: $*" >&2
env OMP_NUM_THREADS=16 "$@"
' gpu.sh "$@"
