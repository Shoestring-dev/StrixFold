"""One build of aux_attn.hip on the 2048-row prompt case, launched twice (for rocprofv3 --pmc counter passes).
Usage: x_attn_one.py block hsaco [rows]."""
import sys

import common as C
import hiprt as H
import t_hipattn as T
import x_hipattn_perf as XP

block, path = int(sys.argv[1]), sys.argv[2]
rows = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
f = H.Module(open(path, "rb").read()).function("tf_ds_aux_sparse_attn_h32_d512")
C.gpu_lock()
d = XP.inputs(rows, True)
keep, captured = [], {}
real = H.launch
def grab(fn, grid, blk, shared, args):
    captured["a"], captured["grid"] = args[0], grid
H.launch = grab
T.hip_attn(d, 1, keep)
H.launch = real
for _ in range(2):
    H.launch(f, captured["grid"], (block, 1, 1), 0, [captured["a"]])
H.sync()
