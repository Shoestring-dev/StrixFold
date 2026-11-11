#!/usr/bin/env bash
# Build an optional HIP vision worker with the same compiler and device libraries as the serving kit.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 ROCM_ROOT GPU_ARCH OUTPUT_DIR" >&2
  exit 2
fi
rocm=$1
arch=$2
out=$3
[[ $arch =~ ^gfx[0-9a-z]+$ ]] || { echo "invalid HIP architecture" >&2; exit 2; }
src=$(cd "$(dirname "$0")/../.." && pwd)/zig/src/families/dsv41
mkdir -p "$out"
"$rocm/bin/amdclang++" -x hip --offload-device-only --no-gpu-bundle-output --offload-arch="$arch" \
  --rocm-path="$rocm" --hip-device-lib-path="$rocm/lib/llvm/amdgcn/bitcode" -O3 \
  "$src/vision_softmax.hip" -o "$out/vision-softmax.co"
cat "$src/vision_hip.py" "$src/vision_cpu.py" > "$out/vision-worker.py"
sha256sum "$out/vision-softmax.co" "$out/vision-worker.py"
