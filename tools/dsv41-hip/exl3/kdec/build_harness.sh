#!/bin/bash
# build_harness.sh: hip/kdec (w-exl3's engine-ABI harness + k-dec's row-invariance modes)
set -euo pipefail
cd ~/tfp/kdec
E=$HOME/tfp/exl3
hipcc -O2 -std=c++20 --offload-arch=gfx1151 -ffp-contract=off -I $E/hip -I $E/oracle -I $HOME/tfp/wt/kdec/zig/kernels/hip/dsv41 \
    hip/kdec.hip -o hip/kdec.new -L $E/oracle -lexl3_oracle -Wl,-rpath,$E/oracle
mv hip/kdec.new hip/kdec; echo "built hip/kdec $(date -Is)"
