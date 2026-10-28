#!/usr/bin/env python3
"""cmp.py A.jsonl[,A2.jsonl...] B.jsonl[,B2...]: per (mode, group / layer, R): A and B medians and back-to-back times
(us; with several files per side, the mean over them) and B / A, plus sclk medians; the summed totals per R."""
import json, sys
from collections import defaultdict
def load(spec):
    d = defaultdict(list)
    for f in spec.split(","):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            m = r.get("mode")
            if m == "glinear": key = ("gl", r["group"], r["rows"])
            elif m == "dexperts": key = ("x", "L%s%s" % (r["layer"], "m" if r.get("k2_gu", [0])[1] >= 12 and r["experts"] < 200 else ""), r["rows"])
            else: continue
            d[key].append((r["median_us"], r["b2b_us"], r["sclk_med"], r.get("pass"), r.get("graph_us", 0.0)))
    return {k: (sum(v[0] for v in vs) / len(vs), sum(v[1] for v in vs) / len(vs), [v[2] for v in vs], all(v[3] for v in vs), sum(v[4] for v in vs) / len(vs)) for k, vs in d.items()}
a, b = load(sys.argv[1]), load(sys.argv[2])
tot = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
for k in sorted(set(a) & set(b), key=lambda k: (k[0], k[2], k[1])):
    A, B = a[k], b[k]
    gr = "  graph %8.1f -> %8.1f (%.3f)" % (A[4], B[4], B[4] / A[4]) if A[4] and B[4] else ""
    print("%-3s %-10s R %2d  med %8.1f -> %8.1f (%.3f)  b2b %8.1f -> %8.1f (%.3f)%s  sclk %s -> %s  pass %s/%s" % (k[0], k[1], k[2], A[0], B[0], B[0] / A[0], A[1], B[1], B[1] / A[1], gr, A[2], B[2], A[3], B[3]))
    t = tot[(k[0], k[2])]
    t[0] += A[0]; t[1] += B[0]; t[2] += A[1]; t[3] += B[1]; t[4] += A[4]; t[5] += B[4]
for k in sorted(tot):
    t = tot[k]
    gr = "  graph %9.1f -> %9.1f (%.3f)" % (t[4], t[5], t[5] / t[4]) if t[4] and t[5] else ""
    print("TOTAL %-3s R %2d  med %9.1f -> %9.1f (%.3f)  b2b %9.1f -> %9.1f (%.3f)%s" % (k[0], k[1], t[0], t[1], t[1] / t[0], t[2], t[3], t[3] / t[2], gr))
