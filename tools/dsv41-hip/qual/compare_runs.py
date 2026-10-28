#!/usr/bin/env python3
"""compare_runs.py A_DIR B_DIR [OUT.json]: token-id equivalence of two served blocks' serve_qual-main.json on the cases
both ran (e.g. eager vs HIP graphs, or ordinary vs DSpark): every reply's (content, token_sha) pair must be equal. Prints
and writes the per-case result; exit 0 only if all compared replies are equal."""
import json
import os
import sys


def replies(d):
    """case -> list of (content, token_sha) in a fixed order, from serve_qual's data blocks"""
    out = {}
    for c in d["cases"]:
        if c["status"] == "skipped":
            continue
        got = []

        def walk(x):
            if isinstance(x, dict):
                if "token_sha" in x and x.get("status") == 200:
                    got.append((x.get("content"), x.get("token_sha")))
                for k in sorted(x):
                    if k not in ("stream_text",):
                        walk(x[k])
            elif isinstance(x, list):
                for y in x:
                    walk(y)
        walk(c["data"])
        out[c["case"]] = got
    return out


a_dir, b_dir = sys.argv[1], sys.argv[2]
A = replies(json.load(open(os.path.join(a_dir, "serve_qual-main.json"))))
B = replies(json.load(open(os.path.join(b_dir, "serve_qual-main.json"))))
res = {}
for case in sorted(set(A) & set(B)):
    a, b = A[case], B[case]
    res[case] = {"replies": [len(a), len(b)], "equal": a == b and len(a) > 0}
doc = {"a": a_dir, "b": b_dir, "cases": res, "all_equal": all(v["equal"] for v in res.values()) and bool(res)}
print(json.dumps(doc, indent=1))
if len(sys.argv) > 3:
    json.dump(doc, open(sys.argv[3], "w"), indent=1)
sys.exit(0 if doc["all_equal"] else 1)
