#!/bin/bash
# build_harness_r.sh: hip/kdec_r from hip/kdec_r.hip (k-dec2 receipt edits), as build_harness.sh
set -euo pipefail
cd ~/tfp/kdec
E=$HOME/tfp/exl3
hipcc -O2 -std=c++20 --offload-arch=gfx1151 -ffp-contract=off -I $E/hip -I $E/oracle -I ${INC:-$HOME/tfp/wt/kdec/zig/kernels/hip/dsv41} \
    hip/kdec_r.hip -o hip/kdec_r.new -L $E/oracle -lexl3_oracle -Wl,-rpath,$E/oracle
mv hip/kdec_r.new hip/kdec_r; echo "built hip/kdec_r $(date -Is)"
