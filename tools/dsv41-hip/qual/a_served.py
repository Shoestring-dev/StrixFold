#!/usr/bin/env python3
"""a_served.py MODE URL OUT [options]: the served requests of the parity checks (PLAN-research-actions.md sections 1 and
4), against rank 0's HTTP API on its own box. Standard library only. Every request and its reply go to OUT (JSON).

  a1 --requests FILE      A1's served check: each request of par_prompts.py's a1 file (but the ref* rows) as a token-id
                          prompt to /v1/completions, greedy, max_tokens 32, in order: the determinism control
                          [A, X, A', X], then [A, B, A', B] for each P_B. Records prompt_tokens, token_sha and ids.
  a4                      A4's sampling checks (/v1/chat/completions, max_tokens 32): greedy; temperature 0.7 with
                          top_k 20, 56, omitted, 0 and 100 (seed 1234 each, so the replies can be compared: equal ids
                          to the explicit 20 or 56 name the effective k); one seed twice and another seed (seeded);
                          the same unseeded request twice (TF_SEED_RANDOM); a greedy request without max_tokens
                          (MAX_TOKENS); a final greedy probe (the server still serves after any failure).
                          --discriminate N: before the probe, N seeds of a 96-token story at temperature 0.7
                          (--discriminate-temperature), each with top_k 20, 56, omitted, 0 and 100 (where 20 and 56
                          differ, the others' effective k shows).
  speed --texts FILE --warm-offset W --offset M
                          one launch of the A2/A3 speed session (section 4): a warm-up 8K prefill (long_prompt.py's
                          document cut at offset W of FILE, discarded), the code and prose decode prompts at 400 tokens
                          (bench_stream.py's requests and metrics), then one 8K prefill at offset M (max_tokens 1,
                          non-streamed: prompt_tokens / the request's wall time, the Spark PP metric).
"""
import argparse
import http.client
import json
import os
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bench_stream  # noqa: E402  (the served decode metric, one definition)

LONG_Q = "Summarise the main events described above in a few sentences."
POEM = "Write a four-line poem about the sea."
STORY = "Write a short story about a lighthouse keeper who finds a message in a bottle."


def post(url, path, body, timeout=1800):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    t0 = time.monotonic()
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    raw = r.read()
    wall = time.monotonic() - t0
    c.close()
    try:
        obj = json.loads(raw)
    except ValueError:
        obj = {"raw": raw.decode("utf-8", "replace")[:2000]}
    tf = obj.get("tensorfold") or {}
    ch = (obj.get("choices") or [{}])[0]
    return {"status": r.status, "wall_s": round(wall, 4), "usage": obj.get("usage"), "finish": ch.get("finish_reason"),
            "text": ch.get("text") if "text" in ch else (ch.get("message") or {}).get("content"),
            "token_sha": tf.get("token_sha"), "ids": tf.get("token_ids"), "sampling": tf.get("sampling"),
            "prefill_seconds": tf.get("prefill_seconds"), "error": obj.get("error")}


def doc(src, offset, chars=45000):
    return ((src + "\n") * 4)[offset:offset + chars]


def mode_a1(a):
    reqs = json.load(open(a.requests))["requests"]
    out = []
    for r in reqs:
        if r["name"].startswith("ref"):
            continue
        res = post(a.url, "/v1/completions", {"model": a.model, "prompt": r["prompt"], "max_tokens": 32, "temperature": 0,
                                               "return_token_ids": True})
        res.update({"name": r["name"], "rows": len(r["prompt"])})
        out.append(res)
        print(json.dumps({k: res[k] for k in ("name", "rows", "status", "token_sha", "wall_s")}), flush=True)
    return {"mode": "a1", "results": out}


def mode_a4(a):
    msgs = [{"role": "user", "content": POEM}]
    cases = [("greedy", {"temperature": 0}),
             ("k20", {"temperature": 0.7, "top_k": 20, "seed": 1234}),
             ("k56", {"temperature": 0.7, "top_k": 56, "seed": 1234}),
             ("k_omitted", {"temperature": 0.7, "seed": 1234}),
             ("k0", {"temperature": 0.7, "top_k": 0, "seed": 1234}),
             ("k100", {"temperature": 0.7, "top_k": 100, "seed": 1234}),
             ("seed777_a", {"temperature": 0.7, "top_k": 20, "seed": 777}),
             ("seed777_b", {"temperature": 0.7, "top_k": 20, "seed": 777}),
             ("seed99", {"temperature": 0.7, "top_k": 20, "seed": 99}),
             ("unseeded_a", {"temperature": 0.7, "top_k": 20}),
             ("unseeded_b", {"temperature": 0.7, "top_k": 20}),
             ("no_max_tokens", {"temperature": 0, "max_tokens": None}),
             ("probe_after", {"temperature": 0})]
    # --discriminate N: N seeds of a longer free-running reply, each with top_k 20, 56, omitted, 0 and 100, so a seed
    # where the explicit 20 and 56 differ shows which one each of the others drew as
    for seed in range(11, 11 + 11 * a.discriminate, 11):
        for k in (20, 56, None, 0, 100):
            cases.insert(-1, (f"d{seed}.k{'_omitted' if k is None else k}",
                              {"messages": [{"role": "user", "content": STORY}], "temperature": a.discriminate_temperature, "top_k": k, "seed": seed, "max_tokens": 96}))
    out = []
    for name, extra in cases:
        body = {"model": a.model, "messages": msgs, "max_tokens": 32, "return_token_ids": True}
        body.update(extra)
        body = {k: v for k, v in body.items() if v is not None}
        res = post(a.url, "/v1/chat/completions", body, timeout=600)
        res.update({"name": name, "request": {k: v for k, v in body.items() if k != "messages"}})
        out.append(res)
        print(json.dumps({k: res[k] for k in ("name", "status", "finish", "token_sha", "error")}), flush=True)
    return {"mode": "a4", "results": out}


def mode_speed(a):
    src = open(a.texts, encoding="utf-8").read()
    out = {"mode": "speed", "warm_offset": a.warm_offset, "offset": a.offset}

    def prefill(offset):
        body = {"model": a.model, "messages": [{"role": "user", "content": doc(src, offset) + "\n\nQuestion: " + LONG_Q}],
                "max_tokens": 1, "temperature": 0}
        res = post(a.url, "/v1/chat/completions", body)
        pt = (res.get("usage") or {}).get("prompt_tokens")
        res["prefill_tok_per_s"] = round(pt / res["wall_s"], 1) if pt and res["status"] == 200 else None
        res["cached_tokens"] = ((res.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        return res

    out["warmup"] = prefill(a.warm_offset)
    print(json.dumps({"warmup": {k: out["warmup"][k] for k in ("status", "wall_s", "prefill_tok_per_s")}}), flush=True)
    for case in ("code", "prose"):
        try:
            out[case] = bench_stream.run(a.url, a.model, bench_stream.PROMPTS[case], 400, 600)
        except Exception as e:  # recorded, the others still run
            out[case] = {"error": repr(e)}
        print(json.dumps({case: {k: out[case].get(k) for k in ("prompt_tokens", "completion_tokens", "decode_tok_per_s", "error")}}), flush=True)
    out["prefill"] = prefill(a.offset)
    print(json.dumps({"prefill": {k: out["prefill"][k] for k in ("status", "wall_s", "prefill_tok_per_s", "cached_tokens")}}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("a1", "a4", "speed"))
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--requests")
    ap.add_argument("--texts")
    ap.add_argument("--warm-offset", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--discriminate", type=int, default=0)
    ap.add_argument("--discriminate-temperature", type=float, default=0.7)
    a = ap.parse_args()
    t0 = time.time()
    res = {"a1": mode_a1, "a4": mode_a4, "speed": mode_speed}[a.mode](a)
    res.update({"url": a.url, "started_unix": round(t0, 1), "seconds": round(time.time() - t0, 1)})
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
