#!/usr/bin/env python3
"""holds_check.py [BASELINE] [M0_PREFLIGHT_JSON]: this box's apt holds against the hold list placed on 2026-10-19 (~11:05
PDT, decision U22: ~/tfp/apt-holds-2026-10-19.txt) and each held package's version against the M0 preflight's.
Prints JSON {ok, held, baseline, differs_from_baseline, version_changed_since_m0, not_on_hold}; exit 0 only if ok."""
import json
import os
import subprocess
import sys

base = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/tfp/apt-holds-2026-10-19.txt")
m0 = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
held = sorted(subprocess.run(["apt-mark", "showhold"], capture_output=True, text=True).stdout.split())
want = sorted(open(base).read().split())
q = subprocess.run(["dpkg-query", "-W", "-f=${Package} ${Version} ${db:Status-Abbrev}\n"] + want, capture_output=True, text=True).stdout
ver = {l.split()[0]: l.split()[1:] for l in q.splitlines() if l.strip()}
ref = (m0.get("packages") or {}).get("gpu_kernel_firmware", {})
changed = {p: [ref[p], v[0]] for p, v in ver.items() if p in ref and ref[p] != v[0]}
not_held = sorted(p for p, v in ver.items() if len(v) < 2 or not v[1].startswith("h"))
out = {"held": len(held), "baseline": len(want), "differs_from_baseline": sorted(set(held) ^ set(want)),
       "version_changed_since_m0": changed, "not_on_hold": not_held, "m0_versions_compared": sum(1 for p in ver if p in ref)}
out["ok"] = not out["differs_from_baseline"] and not changed and not not_held
print(json.dumps(out))
sys.exit(0 if out["ok"] else 1)
