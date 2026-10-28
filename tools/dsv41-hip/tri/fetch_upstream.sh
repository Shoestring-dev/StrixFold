#!/bin/bash
# The pinned inputs of the w-tri port into $TRI_WORK/upstream (default ~/tfp/tri/upstream), from the TensorFold git
# history this worktree shares: the Triton sources at 222a9c0b (deepseek-v41-tp2), the CUDA kernels, the recorded
# launches and the licence at strix-port-base; then their sha256 checked against SHA256SUMS next to this script.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${TRI_WORK:-$HOME/tfp/tri}/upstream
mkdir -p "$out"
tp2=222a9c0b6ecaa9fd3c5dbded65a668524c9510ff
base=58e0171                                    # strix-port-base: 3ff3b2ac + the three TP2 patches (tree 1beea238)
git -C "$here" cat-file -e "$tp2^{commit}"
for f in src/tensorfold/families/deepseek_v41/cuda/kernels.py src/tensorfold/families/deepseek_v41/cuda/markov.py \
         src/tensorfold/cuda/exl3/prefill.py; do
  git -C "$here" show "$tp2:$f" > "$out/$(basename "$f")"
done
git -C "$here" show "$tp2:LICENSE" > "$out/LICENSE"
for f in zig/kernels/cuda/dsv41_ops.cu zig/kernels/cuda/dsv41_torch.cu zig/kernels/cuda/torch_ops/pointwise.cu \
         zig/kernels/cuda/torch_ops/movement.cu zig/kernels/cuda/torch_ops/launch_shape.h \
         zig/src/families/dsv41/fixtures/launches.json; do
  git -C "$here" show "$base:$f" > "$out/$(basename "$f")"
done
(cd "$out" && sha256sum -c "$here/SHA256SUMS")
