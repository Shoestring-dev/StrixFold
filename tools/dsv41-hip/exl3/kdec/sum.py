#!/usr/bin/env python3
"""sum.py FILE...: one line per row-invariance / contract / timing record"""
import json, sys
for f in sys.argv[1:]:
    for l in open(f):
        try: d = json.loads(l)
        except Exception: continue
        m = d.get("mode")
        tag = f.split("/")[-1].replace(".jsonl", "")
        if m in ("rowinv_gl", "rowinv_x"):
            bad = [(b["R"], b["diff"], b["max_ulp"]) for b in d["by_R"] if b["diff"]]
            name = d.get("group", "L%s%s" % (d.get("layer"), " mtp" if d.get("mtp") else ""))
            extra = " max_members %s epoch %s ready_bad %s" % (d.get("max_members"), d.get("epoch"), d.get("ready_bad")) if m == "rowinv_x" else ""
            print("%-18s %-9s %-10s rows %5d diff %8d max_ulp %6d cnt %d%s pass %s %s" % (tag, m, name, d["window_rows"], d["diff"], d["max_ulp"], d["counters_nonzero"], extra, d["pass"], bad[:8]))
        elif m == "glinear":
            c = d["contract"]
            print("%-18s glinear %-9s R %2d med %8.1f us b2b %8.1f GB/s %6.1f/%6.1f sclk %s/%s load %.2f bound %.3f nrmse %.2e bf %d rot %d rope %d pass %s" % (tag, d["group"], d["rows"], d["median_us"], d["b2b_us"], d["gbs_median"], d["gbs_b2b"], d["sclk_min"], d["sclk_med"], d["load1"], c["bound_ratio"], c["nrmse"], d["bf16_y_diff"], d["rot_out_diff"], d["rope_diff"], d["pass"]))
        elif m == "dexperts":
            print("%-18s dexperts L%-3s R %2d places %3d med %8.1f us b2b %8.1f GB/s %6.1f/%6.1f sclk %s/%s load %.2f stages %s nrmse %.2e worst %.2e pass %s" % (tag, d["layer"], d["rows"], d["places"], d["median_us"], d["b2b_us"], d["gbs_median"], d["gbs_b2b"], d["sclk_min"], d["sclk_med"], d["load1"], d["stage_us"], d["nrmse"], d["row_nrmse_max"], d["pass"]))
        else:
            print(tag, l.strip()[:300])
