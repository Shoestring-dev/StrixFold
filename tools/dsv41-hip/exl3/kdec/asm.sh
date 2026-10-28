#!/bin/bash
# asm.sh SRC OUT.s [defs]: the dev copy as gfx1151 assembly (kit flags)
set -euo pipefail
src=$1; out=$2; shift 2
inc=$HOME/tfp/wt/kdec/zig/kernels/hip/dsv41
python3 ~/tfp/kdec/tools/mkdev.py "$src" ~/tfp/kdec/tmp/asm_$$.hip
hipcc --offload-arch=gfx1151 --offload-device-only -S -O3 -std=c++20 -fno-gpu-rdc -mno-wavefrontsize64 -ffp-contract=off -fno-fast-math -fdenormal-fp-math=ieee -D__HIP_PLATFORM_AMD__ "$@" -I "$inc" ~/tfp/kdec/tmp/asm_$$.hip -o "$out"
rm -f ~/tfp/kdec/tmp/asm_$$.hip
