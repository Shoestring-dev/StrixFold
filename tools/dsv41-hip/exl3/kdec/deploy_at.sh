#!/bin/bash
# deploy_at.sh WORKTREE: build that worktree HEAD (native-hip) and deploy it to ~/tfp/deploy-kdec on node0 and node1
set -euo pipefail
W=$1
cd $W
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 1; }
C=$(git rev-parse HEAD)
tools/dsv41-hip/zig_build.sh native-hip -Dhip-triton-kit=$HOME/tfp/tri/kit/gfx1151-v5
tools/dsv41-hip/deploy_rank0.sh --dest "~/tfp/deploy-kdec" --native zig-out/native-hip --bin-commit $C
tools/dsv41-hip/deploy_rank0.sh --host local --dest "~/tfp/deploy-kdec" --native zig-out/native-hip --bin-commit $C
echo "deployed $C"
