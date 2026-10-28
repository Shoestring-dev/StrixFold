#!/usr/bin/env python3
"""serve_verdict.py QOUT CONF MAPS_RC SUITE_RC RESTART RESTART_RC M0_RC M1_RC LOAD_S RESTART_S STRUCT_RC: qual_serve.sh's verdict
(QOUT/qual-serve.json) from the block's files and exit codes; also run by hand to re-judge a finished block."""
import hashlib, json, os, sys
out, conf, maps_rc, suite_rc, restart, restart_rc, m0, m1, load_s, restart_s, struct_rc = sys.argv[1:12]
j = lambda n: json.load(open(os.path.join(out, n))) if os.path.exists(os.path.join(out, n)) else None
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
main, again = j("serve_qual-main.json"), j("serve_qual-restart.json")
restart_equal = None
if restart == "1" and main and again:
    ids = lambda doc, case: [r["data"] for r in doc["cases"] if r["case"] == case]
    pick = lambda d: (d.get("content"), d.get("token_sha")) if isinstance(d, dict) else None
    a = [pick(x) for x in ids(main, "probe")] + [[pick(r) for r in x.get("runs", [])] for x in ids(main, "repeat")]
    b = [pick(x) for x in ids(again, "probe")] + [[pick(r) for r in x.get("runs", [])] for x in ids(again, "repeat")]
    restart_equal = a == b and all(r["status"] == "pass" for r in again["cases"])
struct = j("serve_struct.json")
gates = {
    "structured_output": {"status": ("pass" if struct_rc == "0" else "fail") if struct_rc else "unrun",
                          "cases": struct and ({k: v.get("ok") for k, v in struct["cases"].items()} if isinstance(struct.get("cases"), dict)
                                                    else {c.get("case"): c.get("ok") for c in struct.get("cases", [])})},
    "served_suite": {"status": "pass" if suite_rc == "0" else "fail", "summary": main and main["summary"]},
    "no_nvidia_libraries_mapped": {"status": "pass" if maps_rc == "0" else "fail", "evidence": "maps.txt"},
    "allocation_residency": {"status": "pass" if m0 == "0" and m1 == "0" else "fail",
                             "node0": j("mem-node0.summary.json"), "node1": j("mem-node1.summary.json")},
    "clean_restart": {"status": ("pass" if restart_rc == "0" and restart_equal else "fail") if restart == "1" else "unrun",
                      "token_ids_equal_first_start": restart_equal, "restart_s": restart_s or None},
}
doc = {"schema": "tf-port-qual-serve/1", "run": os.path.basename(out), "config": conf, "config_sha256": sha(conf),
       "rank0": j("resolved-rank0.json"), "rank1": j("resolved-rank1.json"), "load_s": load_s or None,
       "load_before": j("load-before.json"), "load_after": j("load-after.json"), "gates": gates,
       "files": {n: sha(os.path.join(out, n)) for n in sorted(os.listdir(out)) if os.path.isfile(os.path.join(out, n))}}
doc["pass"] = all(g["status"] == "pass" for g in gates.values() if g["status"] != "unrun")
json.dump(doc, open(os.path.join(out, "qual-serve.json"), "w"), indent=1)
print(json.dumps({k: v["status"] for k, v in gates.items()}), "pass" if doc["pass"] else "FAIL")
sys.exit(0 if doc["pass"] else 1)
