#!/usr/bin/env python3
"""mk_r2.py: q-qual's M1.exl3-numeric receipt (receipts/m1_exl3_numeric.json) from q21's results (results/r2_*.jsonl):
every (kernel id, spec, row class) the engine reaches with the DEPLOYED kit's objects, each with the harness records
that graded it against w-ref's contract (fp64 reference, numerics.json exl3.linear / experts gates)."""
import collections, datetime, glob, hashlib, json, os, re

K = os.path.expanduser("~/tfp/kdec")
meta = open(K + "/results/r2.meta").read().split()
dep, xobj, pobj = meta[1], meta[3], meta[5]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
recs = []
for f in sorted(glob.glob(K + "/results/r2_*.jsonl")):
    for line in open(f):
        if line.startswith("{"):
            r = json.loads(line)
            r["_file"] = os.path.basename(f)
            recs.append(r)


def gl_class(m):
    if m == 1: return "1 (decode)"
    if m <= 8: return "2-8 (verify; rows path, one part)"
    if m <= 16: return "9-16 (rows path, two parts)"
    if m < 64: return "17-63 (16-row passes, tail pass)"
    return "64-128 (wide WMMA path)"


def x_class(m):
    if m == 1: return "1"
    if m <= 6: return "2-6 (verify)"
    if m <= 16: return "7-16 (round rows)"
    return "17-63 (short prompt chunks)"


def p_class(m):
    if m <= 128: return "64-128 (prompt chunk tails)"
    if m < 2048: return "129-2047 (prompt chunk tails)"
    return "2048 (full prompt chunk)"


cells = collections.defaultdict(list)   # (kernel id, spec, row class) -> [(pass, ref)]
fails = []
for r in recs:
    m = r.get("mode")
    if m == "glinear":
        ok = bool(r["pass"]) and r.get("counters_nonzero", 0) == 0
        for l in r["layers"]:
            key = ("exl3.glinear", "k2 %d warps %d" % (2 * l["K"], l["wk"]), gl_class(r["rows"]))
            cells[key].append((ok, "%s:%s rows %d (%s)" % (r["_file"], r["group"], r["rows"], l["tensor"])))
        cells[("exl3.rot_many", "-", gl_class(r["rows"]))].append((ok, "%s:%s rows %d" % (r["_file"], r["group"], r["rows"])))
    elif m == "dexperts":
        ok = bool(r["pass"]) and r.get("ready_bad", 0) == 0 and r.get("counters_nonzero", 0) == 0
        hi = max(r["k2_gu"][1], r["k2_d"][1])
        rng = "lo2_hi16" if hi > 10 else "lo2_hi10"
        k2s = sorted(set(r["k2_gu"]) | set(r["k2_d"]))
        for epi in (1, 2):
            cells[("exl3.experts.grouped_cp", "%s epi %d, K2 %s" % (rng, epi, "/".join(map(str, k2s))), x_class(r["rows"]))].append(
                (ok, "%s:layer %s%s rows %d" % (r["_file"], "mtp." if r.get("experts", 0) < 300 else "", r["layer"], r["rows"])))
        cells[("exl3.experts.decode_prep", "-", x_class(r["rows"]))].append((ok, "%s rows %d" % (r["_file"], r["rows"])))
    elif m == "pfexperts":
        ok = bool(r["pass"])
        k2s = r.get("k2_gu_d", [])
        for part in ("count", "place", "rot", "combine"):
            cells[("exl3.pf_experts." + part, "-", p_class(r["rows"]))].append((ok, "%s:layer %d rows %d" % (r["_file"], r["layer"], r["rows"])))
        for part, ks in (("gu", k2s[0]), ("down", k2s[1])):
            for k2 in sorted(set(ks)):
                cells[("exl3.pf_experts." + part, "k2 %d nm %d" % (k2, r["nm"]), p_class(r["rows"]))].append(
                    (ok, "%s:layer %d rows %d" % (r["_file"], r["layer"], r["rows"])))
        gok = r.get("grouped_nrmse", 1) <= 2e-3 and r.get("grouped_row_nrmse_max", 1) <= 8e-3
        for kid in ("exl3.experts.grouped_mma", "exl3.experts.grouped_rows"):
            cells[(kid, "lo2_hi10 (the fallback when pf_experts cannot plan)", p_class(r["rows"]))].append(
                (gok, "%s:layer %d rows %d (same routes)" % (r["_file"], r["layer"], r["rows"])))
    elif m == "pgemm":
        x = r["xgemm"]
        ok = bool(x["contract"]["pass"])
        cells[("exl3.gemm", "k2 %d bm %d" % (2 * r["K"], x["tile"]), p_class(r["rows"]))].append(
            (ok, "%s:%s rows %d" % (r["_file"], r["tensor"], r["rows"])))
        cells[("exl3.rot_in", "-", p_class(r["rows"]))].append((ok, "%s:%s rows %d" % (r["_file"], r["tensor"], r["rows"])))
    else:
        continue
    if not ok:
        fails.append({k: r[k] for k in r if k in ("_file", "mode", "group", "tensor", "layer", "rows", "contract", "nrmse", "row_nrmse_max", "pass")})

matrix = [{"kernel": k[0], "spec": k[1], "rows": k[2], "records": len(v), "pass": all(p for p, _ in v), "evidence": [e for _, e in v][:6]}
          for k, v in sorted(cells.items())]
out = {"schema": "kdec.m1.exl3-numeric/1", "utc": datetime.datetime.utcnow().isoformat() + "Z", "host": os.uname().nodename,
       "deploy": dep, "objects": {"exl3": {"path": xobj, "sha256": sha(xobj)}, "exl3_pf": {"path": pobj, "sha256": sha(pobj)}},
       "harness": "kdec branch tools/dsv41-hip/exl3/kdec/harness.hip (hip/kdec_r), modes glinear / dexperts / pfexperts / pgemm",
       "contract": "w-ref's exl3.linear contract (fp64 reference of the decoded weights; bound ratio, probabilistic exceedances, nrmse <= 1e-3 fp32 out / 3.5e-3 bf16 out) per layer; experts: nrmse <= 2e-3, worst row <= 8e-3, no non-finite, decode scratch protocol clean",
       "not_reached": ["exl3.glinear k2 4 / 6 (no dense tensor of those K2)", "exl3.glinear k2 16 warps 8, k2 8 warps 4 (no such shape)",
                       "exl3.experts.grouped_cp lo8_hi8 (no layer whose experts are all K2 8)",
                       "exl3.pf_experts gu / down nm 128 (needs slots x rows >= 48 x experts: >= 2,640 rows a chunk at 385 experts; chunks are <= 2,048)",
                       "exl3.unpack (the prompt path's gemm reads the trellis directly; unpack is graded bit-exact by the M1.exl3-exact receipt)"],
       "matrix": matrix, "cells": len(matrix), "cells_pass": sum(c["pass"] for c in matrix), "records": len(recs), "failures": fails,
       "pass": all(c["pass"] for c in matrix) and not fails}
json.dump(out, open(K + "/receipts/m1_exl3_numeric.json", "w"), indent=1)
print("cells", out["cells"], "pass", out["cells_pass"], "records", len(recs), "failures", len(fails), "receipt pass", out["pass"])
for c in matrix:
    print("%-34s %-44s %-36s %3d %s" % (c["kernel"], c["spec"], c["rows"], c["records"], "pass" if c["pass"] else "FAIL"))
