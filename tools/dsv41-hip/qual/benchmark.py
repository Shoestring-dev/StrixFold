#!/usr/bin/env python3
"""benchmark.py URL OUT_DIR --prompts prompts-v1.json [--label L] [--reps 5] [--levels 1] [--prefill 2000,8000,16000]
                [--model dsv41] [--draft auto|on|off] [--categories all|LIST]

The port's single-request (and concurrency) benchmark under handoff section 9, on the creator's own prompt set v1
(bench/prompts-v1.json of recipe 9fa230d3, blob 15f511f8; the 8 categories and the counting ceiling, temperature 0,
thinking off, each request tagged at the front so no two share a prefix) through /v1/chat/completions (the port's server
renders the same tokens as DeepSeek's encoder: gate P2.prompt-parity; the creator's TensorFold client sent the
encoder's text to /v1/completions).

Per request, one streamed call with usage and the server's runtime block, and the time (monotonic) of every event that
carries output text. Three decode metrics, each with its numerator and interval frozen here:
  creator     (completion_tokens - 1) / (t_end - t_first): the creator's v41bench_tf.py convention exactly (t_end =
              after the stream's [DONE]; t_first = the first text event); a comparison metric only
  burst       (N_total - N_first_event) / (t_last_event - t_first_event): the handoff's burst-aware client metric;
              N_first_event = the tokens the first event carried: 1 when every event carries one token (events =
              completion tokens), else counted by /v1/tokenize on its text (marked "estimated")
  server      the server's own tokens_per_second (committed tokens over its decode interval), as reported
  request     completion_tokens / t_end: whole-request throughput, TTFT and prefill included
plus TTFT (first text event), total time, completion tokens (usage; reconciled with the committed token ids when the
server returns them), finish reason and the speculative counters (rounds, drafted, accepted, tokens per round).
Prefill: the creator's method (filler words seeded by the target, 0.55 words per target token, max_tokens 1):
prompt tokens / total time, and prompt tokens / TTFT. Each prefill repetition has its own front tag.
Each request's rendered prompt token ids (/v1/tokenize, the same messages and thinking setting) go to
OUT_DIR/prompt_ids.json and must count to usage.prompt_tokens; the run records cached prompt tokens (cold check).
Every cell is repeated --reps times after a warmup of three categories; the report gives median and range. Output:
OUT_DIR/bench-LABEL.json (raw), OUT_DIR/bench-LABEL.md (tables). Standard library only.
"""
import argparse
import hashlib
import http.client
import json
import random
import statistics as st
import sys
import threading
import time
import urllib.parse

A = None


def post(path, body, timeout=1800):
    u = urllib.parse.urlparse(A.url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    c.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return c, c.getresponse()


def ntok_text(text):
    c, r = post("/v1/tokenize", {"model": A.model, "prompt": text})
    d = json.loads(r.read().decode())
    c.close()
    return d.get("count")


PROMPT_IDS = {}   # request tag -> the rendered prompt's token ids (written to OUT/prompt_ids.json with the results)


def prompt_ids(prompt):
    c, r = post("/v1/tokenize", {"model": A.model, "messages": [{"role": "user", "content": prompt}],
                                 "chat_template_kwargs": {"thinking": False, "enable_thinking": False}})
    d = json.loads(r.read().decode())
    c.close()
    return d.get("tokens")


def one(prompt, max_tokens, draft):
    p_ids = prompt_ids(prompt)
    PROMPT_IDS[prompt[:80]] = p_ids
    body = {"model": A.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True,
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    if draft is not None:
        body["draft"] = draft
    t0 = time.monotonic()
    c, r = post("/v1/chat/completions", body)
    events, usage, tf, spec, finish, err = [], None, None, None, None, None
    if r.status != 200:
        err = r.read().decode("utf-8", "replace")[:500]
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
            if ev.get("tensorfold"):
                tf = ev["tensorfold"]
            if ev.get("speculative"):
                spec = ev["speculative"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "") + (ch.get("text") or "")
                if piece:
                    events.append((time.monotonic() - t0, piece))
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.monotonic() - t0
    c.close()
    ct = (usage or {}).get("completion_tokens", 0)
    pt = (usage or {}).get("prompt_tokens", 0)
    ids = (tf or {}).get("token_ids")
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    rec = {"status": r.status, "error": err, "finish": finish, "prompt_tokens": pt, "completion_tokens": ct, "cached_tokens": cached,
           "prompt_ids_sha256": hashlib.sha256(json.dumps(p_ids).encode()).hexdigest() if p_ids is not None else None,
           "prompt_ids_match_usage": p_ids is not None and len(p_ids) == pt,
           "request_tok_s": ct / t_end if t_end > 0 else None,
           "committed_ids": len(ids) if ids is not None else None, "events": len(events), "t_end": round(t_end, 4),
           "t_first": round(events[0][0], 4) if events else None, "t_last": round(events[-1][0], 4) if events else None,
           "server_tok_s": (tf or {}).get("tokens_per_second"), "server_prefill_s": (tf or {}).get("prefill_seconds"),
           "speculative": spec, "text_head": "".join(p for _, p in events)[:120]}
    # metrics with frozen numerators and intervals
    if events and ct > 1 and t_end > events[0][0]:
        rec["creator_tok_s"] = (ct - 1) / (t_end - events[0][0])
    first_n, est = None, False
    if events:
        if len(events) == ct:
            first_n = 1
        else:
            first_n, est = ntok_text(events[0][1]), True
    rec["first_event_tokens"] = first_n
    rec["first_event_tokens_estimated"] = est
    if events and len(events) > 1 and first_n is not None and events[-1][0] > events[0][0]:
        rec["burst_tok_s"] = (ct - first_n) / (events[-1][0] - events[0][0])
    else:
        rec["burst_tok_s"] = None   # undefined: one event, or no interval
    rec["ttft_s"] = rec["t_first"]
    return rec


def batch(c, cat, prompt, max_tokens, tag, draft):
    barrier = threading.Barrier(c)
    out = [None] * c

    def go(i):
        p = f"[bench v1 {tag} {cat} c{c} s{i}] " + prompt
        barrier.wait()
        out[i] = one(p, max_tokens, draft)
    t0 = time.monotonic()
    ts = [threading.Thread(target=go, args=(i,)) for i in range(c)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.monotonic() - t0
    toks = sum(r["completion_tokens"] for r in out)
    return {"c": c, "category": cat, "tag": tag, "wall_s": round(wall, 4), "tokens": toks, "agg_tok_s": toks / wall if wall else None,
            "requests": out}


def filler(n_words, seed, words):
    rnd = random.Random(seed)
    return " ".join(f"{rnd.choice(words)}{rnd.randint(0, 999)}" for _ in range(n_words))


def med_range(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"median": round(st.median(xs), 3), "min": round(min(xs), 3), "max": round(max(xs), 3), "n": len(xs)}


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--levels", default="1")
    ap.add_argument("--prefill", default="")
    ap.add_argument("--prefill-reps", type=int, default=3)
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--draft", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--categories", default="all")
    A = ap.parse_args()
    P = json.load(open(A.prompts))
    prompts_sha = hashlib.sha256(open(A.prompts, "rb").read()).hexdigest()
    draft = None if A.draft == "auto" else (A.draft == "on")
    cats = [(c["name"], c["prompt"], c["max_tokens"]) for c in P["categories"]] + [(P["ceiling"]["name"], P["ceiling"]["prompt"], P["ceiling"]["max_tokens"])]
    if A.categories != "all":
        keep = A.categories.split(",")
        cats = [c for c in cats if c[0] in keep]
    levels = [int(x) for x in A.levels.split(",")]
    res = {"schema": "tf-port-benchmark/1", "label": A.label, "prompt_set": P["version"], "prompts_sha256": prompts_sha, "draft": A.draft,
           "levels": levels, "reps": A.reps, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "conventions": {"creator": "(completion_tokens - 1) / (t_end - t_first)", "burst": "(N_total - N_first_event) / (t_last - t_first)",
                           "server": "tensorfold.tokens_per_second", "prefill": "prompt_tokens / t_end (max_tokens 1); also prompt_tokens / ttft"},
           "warmup": [], "batches": [], "prefill": []}
    for n, p, m in cats[:3]:
        res["warmup"].append(batch(1, n, p, 64, "warm", draft))
    for c in levels:
        for rep in range(A.reps):
            for n, p, m in cats:
                b = batch(c, n, p, m, f"r{rep}", draft)
                res["batches"].append(b)
                r0 = b["requests"][0]
                print(f"C{c} rep{rep} {n:13s} creator {r0.get('creator_tok_s') or 0:6.2f}  burst {r0.get('burst_tok_s') or 0:6.2f}  "
                      f"server {r0.get('server_tok_s') or 0:6.2f}  ttft {r0.get('ttft_s')}  tok {r0['completion_tokens']} {r0['finish']}"
                      + (f"  ERROR {r0['error']}" if r0["error"] else ""), flush=True)
    words = P["prefill_filler"]["words"]
    for tgt in [int(x) for x in A.prefill.split(",") if x]:
        for rep in range(A.prefill_reps):
            r = one(f"[bench v1 prefill {tgt} r{rep}] " + filler(int(tgt * 0.55), tgt, words) + "\n\nReply with the single word OK.", 1, draft)
            row = {"target": tgt, "rep": rep, "prompt_tokens": r["prompt_tokens"], "cached_tokens": r["cached_tokens"], "t_end": r["t_end"], "ttft_s": r["ttft_s"],
                   "prefill_tok_s": r["prompt_tokens"] / r["t_end"] if r["t_end"] else None,
                   "prefill_tok_s_ttft": r["prompt_tokens"] / r["ttft_s"] if r["ttft_s"] else None,
                   "server_prefill_s": r["server_prefill_s"], "status": r["status"], "error": r["error"]}
            res["prefill"].append(row)
            print(f"prefill {tgt} rep{rep}: {row}", flush=True)
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # summary: per level and category, median and range over the repetitions (request 0 of each batch at C1; every
    # request at C > 1); the headline averages the 8 categories (the ceiling apart), as the creator's table does
    summ, eight_of = {}, {}
    for c in levels:
        rows = {}
        for n, _, _ in cats:
            reqs = [r for b in res["batches"] if b["c"] == c and b["category"] == n for r in b["requests"]]
            aggs = [b["agg_tok_s"] for b in res["batches"] if b["c"] == c and b["category"] == n]
            rows[n] = {k: med_range([r.get(k) for r in reqs]) for k in ("creator_tok_s", "burst_tok_s", "server_tok_s", "request_tok_s", "ttft_s", "completion_tokens")}
            rows[n]["prompt_ids_match_usage"] = all(r.get("prompt_ids_match_usage") for r in reqs)
            rows[n]["agg_tok_s"] = med_range(aggs)
            rows[n]["errors"] = sum(1 for r in reqs if r["error"] or r["status"] != 200)
            rows[n]["finish"] = sorted({str(r["finish"]) for r in reqs})
            rows[n]["tokens_reconciled"] = all(r["committed_ids"] in (None, r["completion_tokens"]) for r in reqs)
            rows[n]["cold"] = all(not r.get("cached_tokens") for r in reqs)   # no prompt-cache hit on any repetition
            specs = [r["speculative"] for r in reqs if r.get("speculative")]
            if specs:
                rows[n]["tokens_per_round"] = med_range([s.get("tokens_per_round") for s in specs])
                rows[n]["acceptance_rate"] = med_range([s.get("acceptance_rate") for s in specs])
        eight = [n for n, _, _ in cats if n != P["ceiling"]["name"]]
        head = {k: round(st.mean(rows[n][k]["median"] for n in eight if rows[n].get(k)), 2) if eight and all(rows[n].get(k) for n in eight) else None
                for k in ("creator_tok_s", "burst_tok_s", "server_tok_s", "request_tok_s", "agg_tok_s", "ttft_s")}
        summ[f"C{c}"] = {"categories": rows, "headline_8cat_mean_of_medians": head}
        eight_of[f"C{c}"] = eight
    pre = {}
    for tgt in sorted({p["target"] for p in res["prefill"]}):
        ps = [p for p in res["prefill"] if p["target"] == tgt]
        pre[str(tgt)] = {"prompt_tokens": ps[0]["prompt_tokens"], "prefill_tok_s": med_range([p["prefill_tok_s"] for p in ps]),
                         "prefill_tok_s_ttft": med_range([p["prefill_tok_s_ttft"] for p in ps]), "errors": sum(1 for p in ps if p["error"]),
                         "cold": all(not p.get("cached_tokens") for p in ps)}
    res["summary"] = {"levels": summ, "prefill": pre}
    json.dump(PROMPT_IDS, open(f"{A.out}/prompt_ids.json", "w"))
    res["prompt_ids_sha256"] = hashlib.sha256(open(f"{A.out}/prompt_ids.json", "rb").read()).hexdigest()
    json.dump(res, open(f"{A.out}/bench-{A.label}.json", "w"), indent=1)
    md = [f"## {A.label} ({res['started_utc']})", "", f"Prompt set v1 (sha256 {prompts_sha[:16]}), temperature 0, thinking off, draft {A.draft}, "
          f"{A.reps} repetitions per cell after a 3-request warmup; median [min-max].", ""]
    for lv, s in summ.items():
        md += [f"### {lv}", "", "| category | creator tok/s | burst tok/s | server tok/s | whole request tok/s | TTFT s | tokens | errors |", "|---|---|---|---|---|---|---|---|"]
        f = lambda x: f"{x['median']} [{x['min']}-{x['max']}]" if x else "-"
        for n, r in s["categories"].items():
            md.append(f"| {n} | {f(r['creator_tok_s'])} | {f(r['burst_tok_s'])} | {f(r['server_tok_s'])} | {f(r['request_tok_s'])} | {f(r['ttft_s'])} | {f(r['completion_tokens'])} | {r['errors']} |")
        md += ["", f"Headline ({len(eight_of[lv])} categories, the ceiling apart; mean of medians): {s['headline_8cat_mean_of_medians']}; "
               f"every prompt cold (no cached tokens): {all(r['cold'] for r in s['categories'].values())}", ""]
    if pre:
        md += ["### Prefill", "", "| target | prompt tokens | prefill tok/s (t_end) | prefill tok/s (TTFT) | errors |", "|---|---|---|---|---|"]
        for t, p in pre.items():
            f = lambda x: f"{x['median']} [{x['min']}-{x['max']}] (n {x['n']})" if x else "-"
            md.append(f"| {t} | {p['prompt_tokens']} | {f(p['prefill_tok_s'])} | {f(p['prefill_tok_s_ttft'])} | {p['errors']} |")
    open(f"{A.out}/bench-{A.label}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    errs = sum(r["errors"] for s in summ.values() for r in s["categories"].values()) + sum(p["errors"] for p in pre.values())
    return 0 if errs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
