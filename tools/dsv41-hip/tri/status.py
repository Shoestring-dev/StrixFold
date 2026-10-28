"""Machine-readable per-kernel status from the latest test logs (results/*.jsonl, newest record per (kernel, case)),
plus the kit's register / LDS use: kernel_status.json."""
import json, glob, collections, os
from pathlib import Path
root = Path(os.environ.get("TRI_WORK", Path.home() / "tfp" / "tri"))
recs = {}
for f in sorted(glob.glob(str(root / "results" / "t_*.jsonl"))):
    if "t_perf" in f:
        continue
    for line in open(f):
        d = json.loads(line)
        recs[(Path(f).stem, d.get("kernel"), d.get("case"))] = d
by = collections.defaultdict(lambda: {"checks": 0, "pass": 0, "fail": [], "suites": set(), "worst": {}})
for (suite, k, case), d in recs.items():
    if d.get("ok") is None:
        continue
    b = by[k]
    b["checks"] += 1
    b["suites"].add(suite)
    if d["ok"]:
        b["pass"] += 1
    else:
        b["fail"].append(case)
    for m in ("max_err_over_bound", "max_err_over_allowed", "max_ulp", "frac_not_nearest", "nl2"):
        if isinstance(d.get(m), (int, float)):
            b["worst"][m] = max(b["worst"].get(m, 0), d[m])
kit = json.load(open(root / "kit" / "gfx1151" / "aot.json"))["kernels"]
isa = collections.defaultdict(lambda: {"variants": 0, "vgpr": 0, "spill": 0, "lds": 0, "warps": set()})
for v in kit:
    s = isa[v["fn"]]
    s["variants"] += 1
    s["vgpr"] = max(s["vgpr"], v["isa_stats"].get("vgpr_count", 0))
    s["spill"] = max(s["spill"], v["isa_stats"].get("vgpr_spill_count", 0))
    s["lds"] = max(s["lds"], v["shared"])
    s["warps"].add(v["num_warps"])
out = {}
for k, b in sorted(by.items()):
    out[k] = {"status": "pass" if not b["fail"] else "fail", "checks": b["checks"], "passed": b["pass"],
              "failed_cases": b["fail"], "suites": sorted(b["suites"]), "worst_metrics": b["worst"]}
    if k in isa:
        s = isa[k]
        out[k]["kit"] = {"variants": s["variants"], "max_vgpr": s["vgpr"], "max_vgpr_spill": s["spill"],
                         "max_lds": s["lds"], "num_warps": sorted(s["warps"])}
json.dump(out, open(root / "kernel_status.json", "w"), indent=1)
tot = sum(v["checks"] for v in out.values()); ps = sum(v["passed"] for v in out.values())
print(f"{len(out)} kernels/ops, {tot} checks, {ps} pass, {tot - ps} fail")
for k, v in out.items():
    print(f"{v['status']:4s} {k:44s} {v['passed']:3d}/{v['checks']:<3d} {v.get('kit', {})}")
