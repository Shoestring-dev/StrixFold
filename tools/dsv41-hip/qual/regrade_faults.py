#!/usr/bin/env python3
"""regrade_faults.py OUT_BASE_DIR [--invalid CASE=REASON ...] [--rank0-time-before-ssh]: re-judge a served_faults.sh run
from its saved logs under the current rules, and mark cases a later script fix showed to be invalid. Writes
OUT_BASE_DIR/faults-regraded.json beside faults.json (which stays as it was written).
Rules applied: a peer-mismatch refusal names the setting on any handshake/refusal line (the warning lines included); a
case whose rank 0 refused to load because node0's memory was not free ('do not fit the') is unrun, not a result of the
port (U25); --rank0-time-before-ssh: in runs made before the driver took the fault time on the signalled box, a rank-0
fault's time was read on node1 before the ssh round trip (0.1-0.3 s early), so the token-after-fault check of rank-0
cases is not judged (left out of the verdict, with this reason)."""
import argparse
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("--invalid", action="append", default=[])
ap.add_argument("--rank0-time-before-ssh", action="store_true")
a = ap.parse_args()
doc = json.load(open(os.path.join(a.base, "faults.json")))
invalid = dict(x.split("=", 1) for x in a.invalid)
expect = {"mm_round_rows": "round_rows", "mm_aux": "aux", "mm_streams": "streams", "mm_context": "pool", "mm_kit": "kit"}
out = []
for c in doc["cases"]:
    c = dict(c)
    name = c["case"]
    run = [d for d in os.listdir(a.base) if d.endswith("-" + name) and os.path.isdir(os.path.join(a.base, d))]
    d = os.path.join(a.base, run[0]) if run else None
    text = "".join(open(os.path.join(d, f), errors="replace").read() for f in ("rank0.log", "rank1.log")
                   if d and os.path.exists(os.path.join(d, f)))
    if name in expect and d:
        lines = [l for l in text.splitlines() if re.search(r"refus|mismatch|Mismatch|handshake", l)]
        ck = dict(c["data"].get("checks", {}))
        ck["refusal_names_the_setting"] = any(re.search(expect[name], l) for l in lines)
        c["data"] = dict(c["data"], checks=ck, refusal_lines=lines[:6])
        c["status"] = "pass" if all(ck.values()) else "fail"
        c["detail"] = ", ".join(k for k, v in ck.items() if not v) or "refused before serving: " + next((l for l in lines if re.search(expect[name], l)), "")[:200]
        c["regraded"] = True
    if a.rank0_time_before_ssh and c.get("data", {}).get("rank") == 0 and "at_most_one_token_after_fault" in c["data"].get("checks", {}):
        ck = dict(c["data"]["checks"])
        ck.pop("at_most_one_token_after_fault")
        c["data"] = dict(c["data"], checks=ck, not_judged={"at_most_one_token_after_fault": "fault time read on node1 before the ssh round trip"})
        c["status"] = "pass" if all(ck.values()) else "fail"
        c["detail"] = ", ".join(k for k, v in ck.items() if not v) or "all judged checks"
        c["regraded"] = True
    if "do not fit the" in text:
        c["status"], c["detail"], c["regraded"] = "unrun", "node0 not idle: rank 0 refused to load (memory budget, U25)", True
    if name in invalid:
        c["status"], c["detail"], c["regraded"] = "unrun", "invalid: " + invalid[name], True
    out.append(c)
summary = {s: sum(1 for c in out if c["status"] == s) for s in ("pass", "fail", "skipped", "unrun")}
keep = {k: doc[k] for k in ("deploy_commit", "config", "config_sha256", "identity_note") if k in doc}
json.dump({"schema": "tf-port-served-faults/1", "regraded_from": "faults.json", **keep, "summary": summary, "cases": out},
          open(os.path.join(a.base, "faults-regraded.json"), "w"), indent=1)
print(json.dumps(summary))
for c in out:
    print(f"{c['case']:16s} {c['status']:8s} {c['detail'][:150]}")
