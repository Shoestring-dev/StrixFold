#!/bin/bash
set -euo pipefail
cd ~/tfp/kdec
export INC=$HOME/tfp/wt/kdec-pf/zig/kernels/hip/dsv41
tools/build_dev.sh pf $HOME/tfp/wt/kdec-pf/zig/kernels/hip/dsv41/exl3.hip
grep l2_prefetch tmp/pf.res
tools/build_harness_r.sh
