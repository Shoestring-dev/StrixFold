#!/bin/bash
# build_dev.sh TAG SRC [defs...]: decode-only dev object tmp/TAG.hsaco (mkdev.py) with the kit's flags, resource usage
# in tmp/TAG.res. Run through systemd-run (house rules).
set -euo pipefail
cd ~/tfp/kdec
tag=$1; src=$2; shift 2
inc=${INC:-$HOME/tfp/wt/kdec/zig/kernels/hip/dsv41}
python3 tools/mkdev.py "$src" tmp/$tag.hip
hipcc --offload-arch=gfx1151 --offload-device-only --no-gpu-bundle-output -O3 -std=c++20 -fno-gpu-rdc -mno-wavefrontsize64 \
    -ffp-contract=off -fno-fast-math -fdenormal-fp-math=ieee -D__HIP_PLATFORM_AMD__ "$@" -I "$inc" -Rpass-analysis=kernel-resource-usage \
    tmp/$tag.hip -o tmp/$tag.hsaco 2> tmp/$tag.res.raw || { grep -v remark tmp/$tag.res.raw | head -40; exit 1; }
grep -E "Function Name|VGPRs:|VGPRs Spill|Occupancy|LDS Size|ScratchSize" tmp/$tag.res.raw | sed "s/.*remark: *//; s/ \[-Rpass.*//" | paste - - - - - - > tmp/$tag.res
echo "built tmp/$tag.hsaco $(date -Is)"
