#!/usr/bin/env python3
"""calibrate.py client URL OUT.json [--model NAME] | fit OUT_DIR... : the measured AMD DSpark cost profile (handoff 6E,
gate M4.cost-profile; driven by qual/calibrate.sh).

The scheduler (draft.chooseK / scheduler_costs.Profile) picks each round's draft depth k from ROUND_MS (a round's ms by
its verified rows) and DRAFT_MS. calibrate.sh forces the depth with a profile whose cost is flat up to R rows and
prohibitive above (chooseK then takes the deepest k <= R - 1, since the expected kept tokens grow with k), starts the
server once per R in 1 .. 6 (one stream: the depth limit is 5 drafts), and runs `client` inside quiet windows.
  client  three greedy DSpark replies (code, prose, counting; 200 tokens): per reply the server's rounds, drafted and
          accepted tokens, its decode time (seconds - prefill_seconds); rows per round = drafted / rounds + 1 must equal R
          (else the depth was not forced and the point is rejected); the round time = decode time / rounds
  fit     the per-R medians -> profile.json: ROUND_MS (the measured whole round, the drafter pass included, for 1 .. 6
          rows; rows past 6 use the last entry, as the engine does) and DRAFT_MS 0 (already inside each round); also the
          committed tokens per round and the predicted tok/s of each forced depth on these prompts
Standard library only."""
import argparse
import http.client
import json
import os
import statistics as st
import sys
import time
import urllib.parse

PROMPTS = [("code", "Write a Python class implementing an LRU cache with get and put in O(1), with type hints."),
           ("prose", "Explain to a new manager, in about 200 words, why code review slows teams down at first."),
           ("count", "Count from 1 to 120, separated by commas, nothing else.")]


def client(a):
    u = urllib.parse.urlparse(a.url)
    out = []
    for name, text in PROMPTS:
        for rep in range(a.reps):
            body = {"model": a.model, "messages": [{"role": "user", "content": f"[cal {rep}] " + text}], "max_tokens": 200,
                    "temperature": 0, "draft": True, "chat_template_kwargs": {"enable_thinking": False}}
            c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
            c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = json.loads(c.getresponse().read().decode())
            c.close()
            tf, sp, us = r.get("tensorfold") or {}, r.get("speculative") or {}, r.get("usage") or {}
            rounds = sp.get("rounds") or 0
            dec = (tf.get("seconds") or 0) - (tf.get("prefill_seconds") or 0)
            rec = {"prompt": name, "rep": rep, "tokens": us.get("completion_tokens"), "rounds": rounds, "drafted": sp.get("drafted"),
                   "accepted": sp.get("accepted"), "decode_s": dec,
                   "rows_per_round": (sp.get("drafted") or 0) / rounds + 1 if rounds else None,
                   "round_ms": 1000 * dec / rounds if rounds else None,
                   "tokens_per_round": (us.get("completion_tokens") or 0) / rounds if rounds else None}
            out.append(rec)
            print(json.dumps(rec), flush=True)
    json.dump({"rows_forced": a.rows, "runs": out}, open(a.out, "w"), indent=1)


def fit(a):
    pts = {}
    for d in a.dirs:
        doc = json.load(open(os.path.join(d, "calibrate.json")))
        R = doc["rows_forced"]
        good = [x for x in doc["runs"] if x["rows_per_round"] and abs(x["rows_per_round"] - R) < 0.05 and x["round_ms"]]
        pts[R] = {"n": len(good), "of": len(doc["runs"]), "round_ms": st.median(x["round_ms"] for x in good) if good else None,
                  "tokens_per_round": st.median(x["tokens_per_round"] for x in good) if good else None, "dir": d}
    rows = sorted(r for r in pts if pts[r]["round_ms"])
    if rows != list(range(1, len(rows) + 1)):
        print(json.dumps({"error": "rows not measured contiguously from 1", "points": pts}))
        return 1
    round_ms = [round(pts[r]["round_ms"], 2) for r in rows]
    prof = {"schema": "tf-port-cost-profile/1", "ROUND_MS": ",".join(str(x) for x in round_ms), "DRAFT_MS": "0",
            "meaning": "ROUND_MS[R-1]: the measured whole DSpark round with R verified rows, the drafter pass included (so DRAFT_MS 0); "
                       "rows past the last entry use it, as scheduler_costs.Profile.forwardMs does",
            "points": pts, "predicted_tok_s": {r: round(1000 * pts[r]["tokens_per_round"] / pts[r]["round_ms"], 2) for r in rows},
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(prof, open(a.out, "w"), indent=1)
    print(json.dumps({k: prof[k] for k in ("ROUND_MS", "DRAFT_MS", "predicted_tok_s")}))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("client")
    c.add_argument("url")
    c.add_argument("out")
    c.add_argument("--rows", type=int, required=True)
    c.add_argument("--model", default="dsv41")
    c.add_argument("--reps", type=int, default=2)
    f = sub.add_parser("fit")
    f.add_argument("out")
    f.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    if a.cmd == "client":
        client(a)
        return 0
    return fit(a)


if __name__ == "__main__":
    sys.exit(main())
