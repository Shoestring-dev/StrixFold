#!/usr/bin/env python3
"""trace_verdict.py OUT SET CONF DEPLOY RC1 GRADE_RC SECONDS: check_layers.sh's verdict (OUT/trace.json); also run by hand to re-judge a finished trace."""
import hashlib, json, os, sys
out, fixset, conf, dep, rc1, grc, secs = sys.argv[1:8]
txt = open(os.path.join(out, "grade.txt"), errors="replace").read() if os.path.exists(os.path.join(out, "grade.txt")) else ""
verdict = txt.strip().splitlines()[-1] if txt.strip() else "no grade"
rounds = [json.loads(l) for l in open(os.path.join(out, "rank0.jsonl")) if l.startswith('{"rank": 0, "round"')] if os.path.exists(os.path.join(out, "rank0.jsonl")) else []
errors = [r for r in rounds if "error" in r]
verify = any(r.get("rows", 0) > 1 and "error" not in r for r in rounds)
dep_json = json.load(open(os.path.join(dep, "DEPLOY.json")))
reasons = []
if verdict != "PASS":
    import re
    failing = [f"{m.group(1)} ({m.group(2)} points)" for m in re.finditer(r"^(\w+)\s+n=\s*\d+.*?fail=(\d+)", txt, re.M) if m.group(2) != "0"]
    reasons.append(f"grader: {verdict}" + (f": failing point classes {', '.join(failing)}" if failing else ""))
if errors:
    reasons.append(f"rounds stopped: {[r['error'] for r in errors]}")
if not verify:
    reasons.append("no multirow verification window graded")
# the harness exits 1 when any point differs in bytes from the fixture (always, across vendors): that is not a failure;
# a rank that did not reach its final summary line is
done1 = os.path.exists(os.path.join(out, "rank1.jsonl")) and '"all_equal"' in open(os.path.join(out, "rank1.jsonl")).read().strip().splitlines()[-1:][0] \
    if os.path.exists(os.path.join(out, "rank1.jsonl")) and open(os.path.join(out, "rank1.jsonl")).read().strip() else False
done0 = bool(rounds) and os.path.exists(os.path.join(out, "rank0.jsonl")) and '"all_equal"' in open(os.path.join(out, "rank0.jsonl")).read().strip().splitlines()[-1]
if rc1 not in ("0", "1") or not done1:
    reasons.append(f"rank 1 did not finish (exit {rc1})")
if not done0:
    reasons.append("rank 0 did not finish")
routing = None
try:
    routing = json.load(open(os.path.join(out, "grade.json"))).get("routing")
except Exception:
    pass
doc = {"schema": "tf-port-trace/1", "set": fixset, "deploy_commit": dep_json.get("commit"), "config": conf,
       "layer_extra": os.environ.get("LAYER_EXTRA", ""),
       "config_sha256": hashlib.sha256(open(conf, "rb").read()).hexdigest(), "seconds": int(secs),
       "status": "pass" if not reasons else "fail", "reason": "; ".join(reasons) or "grader PASS on every step incl. the verify window",
       "summary": {"verdict": verdict, "rounds": rounds, "routing": {k: v for k, v in (routing or {}).items() if k != "violation_detail"}},
       "grader": open(os.path.join(out, "grader.sha256")).read().split("\n") if os.path.exists(os.path.join(out, "grader.sha256")) else None}
json.dump(doc, open(os.path.join(out, "trace.json"), "w"), indent=1)
print(json.dumps({"status": doc["status"], "reason": doc["reason"]}))
