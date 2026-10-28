#!/usr/bin/env python3
"""mkdev.py SRC OUT: a decode-only dev copy of exl3.hip (no exl3.gemm, no prompt grouped_mma / grouped_rows): the
decode kernels' source is unchanged, only instantiations are dropped, so the dev object compiles in minutes."""
import re, sys
src = open(sys.argv[1]).read().split("\n")
out, skip = [], False
for line in src:
    if re.match(r"^TF_DS_EXL3_GEMM\(", line):
        continue
    if "tf_ds_exl3_experts_grouped_mma_lo" in line or "tf_ds_exl3_experts_grouped_rows_lo" in line:
        skip = True
    if skip and "tf_ds_exl3_experts_grouped_cp_lo" in line:
        skip = False
    if skip:
        continue
    out.append(line)
open(sys.argv[2], "w").write("\n".join(out))
