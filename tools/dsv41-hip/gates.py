#!/usr/bin/env python3
"""gates.py check | summary | render PORT_STATUS.md: the port's gate registry (gates.json) and unknowns register.

gates.json is the one machine-readable registry: every mandatory gate of the handoff and PLAN.md (id, milestone, owner,
status pass/fail/unrun/skipped, evidence paths, date, notes, decision) and the single unknowns register (id, item,
owner, discovery task, due phase, status open/mitigated/noted/closed). Edit gates.json by hand, then:
  check    validate it: unique ids, known statuses, a date and evidence for every executed gate, no unrun gate with a
           date, every unknown referenced by a gate exists, every open unknown has an owner and a task;
  summary  counts by milestone and status;
  render   rewrite the tables between the <!-- gates:begin/end --> and <!-- unknowns:begin/end --> markers of
           PORT_STATUS.md from gates.json, so the status page and the registry cannot disagree.
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
REG = os.path.join(HERE, "gates.json")


def check(reg):
    errs = []
    gs, us = reg["gates"], reg["unknowns"]
    gok, uok = set(reg["status_values"]["gates"]), set(reg["status_values"]["unknowns"])
    uids = {u["id"] for u in us}
    for kind, items in (("gate", gs), ("unknown", us)):
        dup = [k for k, n in collections.Counter(i["id"] for i in items).items() if n > 1]
        errs += [f"duplicate {kind} id {d}" for d in dup]
    for g in gs:
        for k in ("id", "milestone", "title", "source", "owner", "status", "evidence"):
            if k not in g:
                errs.append(f"{g.get('id')}: missing {k}")
        if g.get("status") not in gok:
            errs.append(f"{g['id']}: status {g.get('status')!r}")
        if g.get("status") in ("pass", "fail", "skipped") and (not g.get("date") or not g.get("evidence")):
            errs.append(f"{g['id']}: executed gate without date or evidence")
        if g.get("status") == "unrun" and g.get("date"):
            errs.append(f"{g['id']}: unrun gate with a date")
        errs += [f"{g['id']}: unknown {x} not in the register" for x in g.get("unknowns", []) if x not in uids]
    for u in us:
        if u.get("status") not in uok:
            errs.append(f"{u['id']}: status {u.get('status')!r}")
        if u.get("status") != "closed" and not (u.get("owner") and u.get("discovery_task") and u.get("due")):
            errs.append(f"{u['id']}: open unknown without owner, task or due phase")
    return errs


def cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def tables(reg):
    g = ["| Gate | Milestone | Status | Owner | Evidence | Notes |", "|---|---|---|---|---|---|"]
    for x in reg["gates"]:
        note = x.get("notes", "")
        if x.get("decision"):
            note = (note + " Decision: " + x["decision"]).strip()
        g.append(f"| `{x['id']}` {cell(x['title'])} | {x['milestone']} | **{x['status']}**{' ' + x['date'] if x.get('date') else ''} | "
                 f"{cell(x['owner'])} | {cell('; '.join(x['evidence'])) or '-'} | {cell(note) or '-'} |")
    u = ["| Id | Unknown | Owner | Discovery task | Due | Status |", "|---|---|---|---|---|---|"]
    for x in reg["unknowns"]:
        u.append(f"| {x['id']} | {cell(x['item'])} | {cell(x['owner'])} | {cell(x['discovery_task'])} | {cell(x['due'])} | {x['status']} |")
    return "\n".join(g), "\n".join(u)


def main():
    reg = json.load(open(REG))
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    errs = check(reg)
    if cmd == "check":
        print("\n".join(errs) or f"ok: {len(reg['gates'])} gates, {len(reg['unknowns'])} unknowns")
        return 1 if errs else 0
    if errs:
        print("\n".join(errs))
        return 1
    if cmd == "summary":
        c = collections.Counter((g["milestone"], g["status"]) for g in reg["gates"])
        for m in dict.fromkeys(g["milestone"] for g in reg["gates"]):
            print(m, {s: c[(m, s)] for s in reg["status_values"]["gates"] if c[(m, s)]})
        print("unknowns", dict(collections.Counter(u["status"] for u in reg["unknowns"])))
        return 0
    if cmd == "render":
        path = sys.argv[2]
        text = open(path).read()
        gt, ut = tables(reg)
        for tag, body in (("gates", gt), ("unknowns", ut)):
            text, n = re.subn(rf"(<!-- {tag}:begin -->\n)(?:.*?\n)?(<!-- {tag}:end -->)", lambda m: m.group(1) + body + "\n" + m.group(2), text, flags=re.S)
            if n != 1:
                print(f"{path}: no <!-- {tag}:begin/end --> markers")
                return 1
        open(path, "w").write(text)
        print(f"rendered {len(reg['gates'])} gates and {len(reg['unknowns'])} unknowns into {path}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
