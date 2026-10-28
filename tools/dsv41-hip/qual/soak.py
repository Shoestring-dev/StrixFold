#!/usr/bin/env python3
"""soak.py URL OUT_DIR --prompts prompts-v1.json --seconds 900 [--model dsv41] [--long-every 0] [--long-words 4400]

The served soak's client (handoff M5: a sustained workload long enough to expose thermal and memory trends): back-to-back
streamed greedy requests over the creator's v1 categories (each with a unique front tag), and with --long-every N a
long prompt (about --long-words filler words) every N-th request, until --seconds have passed. Every request's outcome
goes to OUT_DIR/soak.jsonl as it ends (status, finish, tokens, TTFT, decode tok/s (completion_tokens - 1) / (t_end -
t_first), wall, error); OUT_DIR/soak-summary.json gives the counts, errors, and per-minute tokens and decode rate so a
trend (thermal throttling, memory growth slowing the box) shows. Standard library only."""
import argparse
import http.client
import json
import random
import statistics as st
import sys
import time
import urllib.parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--seconds", type=int, default=900)
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--long-every", type=int, default=0)
    ap.add_argument("--long-words", type=int, default=4400)
    a = ap.parse_args()
    P = json.load(open(a.prompts))
    cats = [(c["name"], c["prompt"], c["max_tokens"]) for c in P["categories"]]
    words = P["prefill_filler"]["words"]
    u = urllib.parse.urlparse(a.url)
    log = open(f"{a.out}/soak.jsonl", "w", buffering=1)
    start = time.monotonic()
    wall0 = time.time()
    i = 0
    rows = []
    while time.monotonic() - start < a.seconds:
        long_req = a.long_every and i % a.long_every == a.long_every - 1
        if long_req:
            rnd = random.Random(i)
            name, prompt, mt = "long", " ".join(f"{rnd.choice(words)}{rnd.randint(0, 999)}" for _ in range(a.long_words)) + \
                "\n\nIn one sentence, which word appears most often above?", 64
        else:
            name, prompt, mt = cats[i % len(cats)]
        body = {"model": a.model, "messages": [{"role": "user", "content": f"[soak {i}] " + prompt}], "max_tokens": mt, "temperature": 0,
                "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.monotonic()
        rec = {"i": i, "category": name, "t_start": round(time.time() - wall0, 3)}
        first, usage, finish, err = None, None, None, None
        try:
            c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
            c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = c.getresponse()
            rec["status"] = r.status
            if r.status != 200:
                err = r.read().decode("utf-8", "replace")[:300]
            else:
                for raw in r:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ev = json.loads(data)
                    if ev.get("error"):
                        err = ev["error"]
                    if ev.get("usage"):
                        usage = ev["usage"]
                    for ch in ev.get("choices") or []:
                        d = ch.get("delta") or {}
                        if (d.get("content") or d.get("reasoning_content")) and first is None:
                            first = time.monotonic() - t0
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
            c.close()
        except Exception as e:  # a request that breaks is an error of the soak, never skipped
            err = f"{type(e).__name__}: {e}"
            rec.setdefault("status", None)
        t_end = time.monotonic() - t0
        ct = (usage or {}).get("completion_tokens", 0)
        rec.update({"finish": finish, "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": ct, "ttft_s": first,
                    "t_end": round(t_end, 3), "decode_tok_s": (ct - 1) / (t_end - first) if first is not None and ct > 1 and t_end > first else None,
                    "error": err})
        rec["ok"] = rec.get("status") == 200 and not err and finish in ("stop", "length")
        log.write(json.dumps(rec) + "\n")
        rows.append(rec)
        i += 1
    dur = time.monotonic() - start
    per_min = {}
    for r in rows:
        m = int(r["t_start"] // 60)
        b = per_min.setdefault(m, {"requests": 0, "tokens": 0, "errors": 0, "decode": []})
        b["requests"] += 1
        b["tokens"] += r["completion_tokens"] or 0
        b["errors"] += 0 if r["ok"] else 1
        if r["decode_tok_s"] and r["category"] != "long":
            b["decode"].append(r["decode_tok_s"])
    trend = [{"minute": m, "requests": b["requests"], "tokens": b["tokens"], "errors": b["errors"],
              "decode_median": round(st.median(b["decode"]), 2) if b["decode"] else None} for m, b in sorted(per_min.items())]
    meds = [t["decode_median"] for t in trend if t["decode_median"]]
    summ = {"seconds": round(dur, 1), "requests": len(rows), "errors": sum(1 for r in rows if not r["ok"]),
            "long_requests": sum(1 for r in rows if r["category"] == "long"),
            "tokens": sum(r["completion_tokens"] or 0 for r in rows), "trend": trend,
            "decode_first_vs_last_minute": [meds[0], meds[-1]] if meds else None,
            "first_errors": [r for r in rows if not r["ok"]][:5]}
    json.dump(summ, open(f"{a.out}/soak-summary.json", "w"), indent=1)
    print(json.dumps({k: summ[k] for k in ("seconds", "requests", "errors", "long_requests", "tokens", "decode_first_vs_last_minute")}))
    return 0 if summ["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
