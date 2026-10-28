#!/usr/bin/env python3
"""trace_localize.py KEY... [--dump DIR] [--fix DIR]: for layer-gate points of a check_layers.sh run, the GPU dump against
w-ref's FP64 fixture by mHC stream and by row quarter: the least-squares scale GPU/reference and the nrmse, both ranks.
Used to localize a failing point class (e.g. 0:39:Ref.layer_out.h). Runs on node0 with w-ref's venv (numpy).
  ~/venvs/ref/bin/python trace_localize.py 0:39:Ref.layer_out.h 0:38:Ref.layer_out.h --dump ~/tfp/qual/trace/TAG/dump"""
import argparse
import json
import os
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("keys", nargs="+")
ap.add_argument("--dump", default=os.path.expanduser("~/tfp/qual/trace/p160-58fd/dump"))
ap.add_argument("--fix", default=os.path.expanduser("~/tfp/ref/out/gpufix2-p160-rounds"))
A = ap.parse_args()
D, F = A.dump, A.fix
def lines(d):
    out = {}
    for r in ("rank0", "rank1"):
        p = os.path.join(d, r, "layers.jsonl")
        if os.path.exists(p):
            for l in open(p):
                x = json.loads(l); out[(x["key"], x["rank"])] = x
    return out
def load(d, x):
    a = np.fromfile(os.path.join(d, "layers", x["file"]), dtype=np.uint16 if x["dtype"] == "bfloat16" else np.float32)
    if x["dtype"] == "bfloat16":
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a.reshape(x["shape"]).astype(np.float64)
g, f = lines(D), lines(F)
for key in A.keys:
    for rank in (0, 1):
        if (key, rank) not in g or (key, rank) not in f:
            print(key, rank, "missing", (key, rank) in g, (key, rank) in f); continue
        gx, fx = g[(key, rank)], f[(key, rank)]
        G = load(D, gx); R = load(F, fx)
        rs = gx.get("row_start", 0)
        if R.shape[0] != G.shape[0]:
            R = R[rs:rs + G.shape[0]]
        def scale(a, b):
            return float((a * b).sum() / (b * b).sum())
        streams = [round(scale(G[:, s], R[:, s]), 5) for s in range(G.shape[1])] if G.ndim == 3 else []
        rows = G.shape[0]
        groups = [(0, rows // 4), (rows // 4, rows // 2), (rows // 2, 3 * rows // 4), (3 * rows // 4, rows)]
        print(key, "rank", rank, "shape", G.shape, "row_start", rs, "scale all", round(scale(G, R), 5), "per stream", streams,
              "per row quarter", [round(scale(G[a:b], R[a:b]), 5) for a, b in groups],
              "nrmse", round(float(np.sqrt(((G - R) ** 2).sum() / (R ** 2).sum())), 5))
