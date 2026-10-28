"""A markdown table of the kit's kernels: runtime parameters, constexprs, variants, warps, LDS, registers."""
import json, sys
k = json.load(open(sys.argv[1]))["kernels"]
by = {}
for v in k:
    by.setdefault(v["fn"], []).append(v)
print("| Kernel | Runtime parameters (type) | Constexprs | AMD variants | warps | LDS B | max VGPR / spills |")
print("|---|---|---|---|---|---|---|")
for fn in sorted(by):
    vs = by[fn]
    params = {}
    for v in vs:
        for p in v["params"]:
            params.setdefault(p["name"], set()).add(p["type"])
    cs = sorted(set(c for v in vs for c in v["consts"]))
    ps = ", ".join(n + ":" + "/".join(sorted(t)) for n, t in params.items())
    w = sorted(set(v["num_warps"] for v in vs))
    lds = max(v["shared"] for v in vs)
    vg = max(v["isa_stats"].get("vgpr_count", 0) for v in vs)
    sp = max(v["isa_stats"].get("vgpr_spill_count", 0) for v in vs)
    print(f"| `{fn}` | {ps} | {', '.join(cs)} | {len(vs)} | {w} | {lds} | {vg} / {sp} |")
