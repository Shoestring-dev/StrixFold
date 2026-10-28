#!/usr/bin/env python3
"""spark_compare.py OUT.md BENCH.json [BENCH2.json ...] --spark bench-tp2a.json: our benchmark.py results against the
creator's two-Spark run (recipe 9fa230d3 runs/2026-10-18-1017-tensorfold-tp2/results/tp2a/bench-tp2a.json, git blob
06dd8054), in the Spark's own metric, one table per run:
  per category (C1): our creator-metric decode tok/s (median [min-max] over the repetitions) beside Spark's per-stream
  tok/s and the ratio; our whole-request tok/s beside Spark's C1 aggregate (tokens / batch wall); TTFT; output tokens;
  the 8-category mean (the counting ceiling apart), as the creator's headline;
  prefill: our prompt tokens / whole request (one output token) beside Spark's for the same filler target, and the ratio.
Also writes OUT.json with the same numbers."""
import argparse
import json
import statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("bench", nargs="+")
    ap.add_argument("--spark", required=True)
    a = ap.parse_args()
    sp = json.load(open(a.spark))
    sc = {b["category"]: b for b in sp["batches"] if b["c"] == 1}
    spre = {p["target"]: p for p in sp["prefill"]}
    md, doc = [], {"spark": {"file": a.spark, "headline": sp["headline"][0]}, "runs": []}
    for path in a.bench:
        b = json.load(open(path))
        cats = b["summary"]["levels"]["C1"]["categories"]
        rows, ratios = [], []
        md += [f"## {b['label']} (draft {b['draft']}, {b['reps']} repetitions, prompt set {b['prompt_set']})", "",
               "| category | ours decode tok/s (creator metric) | Spark | ratio | ours whole request tok/s | Spark C1 aggregate | ours TTFT s | Spark TTFT s | ours / Spark tokens |",
               "|---|---|---|---|---|---|---|---|---|"]
        for name, r in cats.items():
            s = sc.get(name)
            ours = r["creator_tok_s"]
            ratio = ours["median"] / s["per_stream_tok_s"] if ours and s else None
            if ratio and name != "ceiling_count":
                ratios.append(ratio)
            f = lambda x: f"{x['median']:.2f} [{x['min']:.2f}-{x['max']:.2f}]" if x else "-"
            md.append(f"| {name} | {f(ours)} | {s['per_stream_tok_s'] if s else '-'} | {ratio:.2f} | {f(r.get('request_tok_s'))} | "
                      f"{s['agg_tok_s'] if s else '-'} | {f(r['ttft_s'])} | {s['ttft_mean_s'] if s else '-'} | "
                      f"{r['completion_tokens']['median'] if r['completion_tokens'] else '-'} / {s['requests'][0]['completion_tokens'] if s else '-'} |"
                      if ratio is not None else f"| {name} | {f(ours)} | {s['per_stream_tok_s'] if s else '-'} | - | | | | | |")
            rows.append({"category": name, "ours": ours, "spark": s and s["per_stream_tok_s"], "ratio": ratio,
                         "ours_request_tok_s": r.get("request_tok_s"), "spark_agg_tok_s": s and s["agg_tok_s"]})
        eight = [x for x in rows if x["category"] != "ceiling_count" and x["ours"]]
        mean_ours = st.mean(x["ours"]["median"] for x in eight) if eight else None
        mean_sp = st.mean(x["spark"] for x in eight if x["spark"]) if eight else None
        md += ["", f"{len(eight)}-category mean: ours {mean_ours:.2f} tok/s, Spark {mean_sp:.2f} (its headline per-stream 84.16), ratio "
               f"{mean_ours / mean_sp:.2f}; per-category ratios {min(ratios):.2f}-{max(ratios):.2f}" if eight and ratios else "", ""]
        pre = b["summary"].get("prefill", {})
        prows = []
        if pre:
            md += ["| prefill target | ours prompt tokens | ours tok/s (whole request) | Spark prompt tokens | Spark tok/s | ratio |", "|---|---|---|---|---|---|"]
            for t, p in pre.items():
                s = spre.get(int(t))
                ours = p["prefill_tok_s"]
                ratio = ours["median"] / s["prefill_tok_s"] if ours and s else None
                md.append(f"| {t} | {p['prompt_tokens']} | {ours['median']:.1f} [{ours['min']:.1f}-{ours['max']:.1f}] | "
                          f"{s['prompt_tokens'] if s else '-'} | {s['prefill_tok_s'] if s else '-'} | {ratio:.2f} |" if ratio else f"| {t} | {p['prompt_tokens']} | - | | | |")
                prows.append({"target": int(t), "ours_prompt_tokens": p["prompt_tokens"], "ours": ours, "spark": s, "ratio": ratio})
            md.append("")
        doc["runs"].append({"bench": path, "label": b["label"], "draft": b["draft"], "categories": rows, "mean_ours": mean_ours,
                            "mean_spark": mean_sp, "prefill": prows})
    open(a.out, "w").write("\n".join(md) + "\n")
    json.dump(doc, open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
