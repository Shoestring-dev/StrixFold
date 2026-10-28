#!/usr/bin/env python3
"""serve_qual.py URL OUT.json [--model NAME] [--context N] [--long 0|1] [--drafts 0|1] [--cases a,b,...]

The served functional suite of qualify.sh (handoff sections 7 and 10): every case talks to the real server (rank 0's
HTTP API) and checks its behaviour, not the prose. Standard library only; runs on rank 0's box against 127.0.0.1.

Cases (each: ok true / false, or null with a reason when it cannot run in this configuration):
  health        /health and /v1/models answer and name the model
  probe         greedy answer contains "Paris"; token ids returned and their count equals usage.completion_tokens
  repeat        the same greedy request three times: identical token ids and finish reason (repeated-request equivalence)
  stream_equal  the same request streamed: the assembled text equals the non-streamed reply's, usage counts agree
  short         every prompt size from the shortest chat prompt to 20 rows (upstream #16's class; each row count has
                its own kernel variants): each completes with a finish reason. The other cases carry a system
                preamble, so their prompts are longer than 16 rows
  system_multi  a system message and a multi-turn conversation complete; the system instruction is followed
  tools         a tool offered (tool_choice auto) and asked for: finish_reason tool_calls, arguments parse as JSON
  json          response_format json_object: the reply parses as JSON
  thinking      enable_thinking: reasoning_content present and content separate; disabled: no reasoning_content
  stop          a stop string ends the reply (finish_reason stop, the string not in the content)
  max_tokens    max_tokens 5: finish_reason length and exactly 5 completion tokens
  capacity      max_tokens = context - prompt accepted; one more refused with a context-length error (400)
  malformed     bad JSON, missing messages, unknown role, negative max_tokens, an image part: 4xx with an error body,
                and the server still answers the probe afterwards
  seeded        temperature 0.8 with a seed: the same seed twice (and streamed) gives the same token ids, another seed
                another reply
  concurrent    two requests at once: both complete and each equals its isolated reply (token ids)
  cancel        a streamed request dropped after a few tokens, then the same request in full: equal to the isolated
                reply; the server keeps serving
  long_prompt   (--long 1) prompts across the 128-row and 2,048-row chunk boundaries, 4K, 8K, 16K (16K pool), and a
                fill to 16 positions under capacity
  replay_boundary (--long 1) A (2,049 rows: a 1-row final chunk) then an unrelated B; B repeated equal
  replay_fresh  (restart suite, --reference) B as a fresh server's first request equals B after A (no stale state)
  drafts        (--drafts 1) per-request ordinary vs DSpark, greedy and seeded (t 0.8): identical token ids, proposals
                and verifications seen, some drafts accepted
  drafts_budget / drafts_eos / drafts_cancel / drafts_accounting  (--drafts 1) max_tokens truncation inside a round, EOS
                inside a proposal, cancellation, and the committed-token accounting, each against the run without drafts
"""
import argparse
import hashlib
import http.client
import json
import sys
import threading
import time
import urllib.parse

A = None
TIMEOUT = 900


def conn():
    u = urllib.parse.urlparse(A.url)
    return http.client.HTTPConnection(u.hostname, u.port, timeout=TIMEOUT)


def request(method, path, body=None, raw=None):
    c = conn()
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    c.request(method, path, body=data, headers={"Content-Type": "application/json"} if data is not None else {})
    r = c.getresponse()
    text = r.read().decode("utf-8", "replace")
    c.close()
    try:
        return r.status, json.loads(text)
    except ValueError:
        return r.status, text


def chat_body(messages, **kw):
    b = {"model": A.model, "messages": messages, "temperature": 0, "max_tokens": 64, "return_token_ids": True}
    b.update(kw)
    return b


def chat(messages, **kw):
    t0 = time.monotonic()
    st, r = request("POST", "/v1/chat/completions", chat_body(messages, **kw))
    wall = time.monotonic() - t0
    if st != 200 or not isinstance(r, dict):
        return {"status": st, "error": r, "wall_s": wall}
    ch = r["choices"][0]
    tf = r.get("tensorfold") or {}
    return {"status": st, "wall_s": round(wall, 3), "finish": ch.get("finish_reason"), "content": ch["message"].get("content"),
            "reasoning": ch["message"].get("reasoning_content"), "tool_calls": ch["message"].get("tool_calls"),
            "usage": r.get("usage"), "ids": tf.get("token_ids"), "token_sha": tf.get("token_sha"),
            "speculative": r.get("speculative"), "prefill_seconds": tf.get("prefill_seconds"),
            "ttft": tf.get("time_to_first_token"), "drafts": tf.get("drafts")}


def stream(messages, cancel_after=None, **kw):
    """A streamed request; with cancel_after=N the connection is closed after N content events."""
    b = chat_body(messages, stream=True, stream_options={"include_usage": True}, **kw)
    c = conn()
    t0 = time.monotonic()
    c.request("POST", "/v1/chat/completions", body=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    out = {"status": r.status, "events": 0, "text": [], "reasoning": [], "finish": None, "usage": None, "tensorfold": None,
           "error": None, "cancelled": False, "first_s": None, "last_s": None}
    if r.status != 200:
        out["error"] = r.read().decode("utf-8", "replace")
        c.close()
        return out
    try:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("error"):
                out["error"] = obj["error"]
            if obj.get("usage"):
                out["usage"] = obj["usage"]
            if obj.get("tensorfold"):
                out["tensorfold"] = obj["tensorfold"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    now = time.monotonic() - t0
                    out["events"] += 1
                    out["first_s"] = out["first_s"] or now
                    out["last_s"] = now
                    (out["text"] if d.get("content") else out["reasoning"]).append(d.get("content") or d.get("reasoning_content"))
                if ch.get("finish_reason"):
                    out["finish"] = ch["finish_reason"]
            if cancel_after is not None and out["events"] >= cancel_after:
                out["cancelled"] = True
                break
    except (http.client.IncompleteRead, ConnectionError, OSError) as e:
        out["error"] = out["error"] or f"{type(e).__name__}: {e}"
    finally:
        c.close()
    out["text"] = "".join(out["text"])
    out["reasoning"] = "".join(out["reasoning"])
    return out


def ntok(messages, **kw):
    st, r = request("POST", "/v1/tokenize", {"model": A.model, "messages": messages, **kw})
    return r.get("count") if st == 200 and isinstance(r, dict) else None


SYS = "You are a careful assistant. Answer accurately and concisely, in plain English, without preamble."


def user(text, system=True):
    """A one-turn conversation. With the system preamble every case's prompt has more than 16 rows, so the short case
    alone covers the 1-16 row prompt sizes (their kernel variants differ from longer prompts')."""
    return ([{"role": "system", "content": SYS}] if system else []) + [{"role": "user", "content": text}]


def brief(r):
    keep = ("status", "finish", "content", "reasoning", "tool_calls", "usage", "token_sha", "speculative", "error", "wall_s")
    return {k: (v[:400] if isinstance(v, str) else v) for k, v in r.items() if k in keep and v is not None}


PROBE = user("Answer in one sentence: what is the capital of France?")


def c_health():
    st, h = request("GET", "/health")
    st2, m = request("GET", "/v1/models")
    ok = st == 200 and isinstance(h, dict) and h.get("status") == "ok" and st2 == 200 and A.model in json.dumps(m)
    return ok, "health ok and the model listed" if ok else "health or models wrong", {"health": h, "models": m}


def c_probe():
    r = chat(PROBE)
    ids = r.get("ids") or []
    ok = r["status"] == 200 and "paris" in (r.get("content") or "").lower() and len(ids) == (r.get("usage") or {}).get("completion_tokens")
    return ok, f"reply {r.get('content')!r}, {len(ids)} ids", brief(r)


def c_repeat():
    m = user("List the first ten prime numbers, comma separated, and nothing else.")
    rs = [chat(m, max_tokens=48) for _ in range(3)]
    sig = {(tuple(r.get("ids") or []), r.get("finish")) for r in rs}
    ok = all(r["status"] == 200 for r in rs) and len(sig) == 1 and len(rs[0].get("ids") or []) > 0
    return ok, f"3 runs, {len(sig)} distinct (ids, finish)", {"runs": [brief(r) for r in rs]}


def c_stream_equal():
    m = user("Write two sentences about the sea.")
    a = chat(m, max_tokens=80)
    s = stream(m, max_tokens=80)
    ok = a["status"] == 200 and s["status"] == 200 and not s["error"] and s["text"] == (a.get("content") or "") \
        and (s["usage"] or {}).get("completion_tokens") == (a.get("usage") or {}).get("completion_tokens") and s["finish"] == a["finish"]
    return ok, f"stream {len(s['text'])} chars / {s['events']} events vs whole {len(a.get('content') or '')} chars", \
        {"whole": brief(a), "stream": {k: s[k] for k in ("status", "events", "finish", "usage", "error")}, "stream_text": s["text"][:400]}


def c_short():
    """Every prompt size from the shortest chat prompt to 20 rows (upstream #16's class, and each row count's kernel
    variants): each must complete with a finish reason."""
    res = []
    base = ntok(user("x", system=False))
    for n in range(base, 21):
        m = filler(n, system=False)
        if m is None:
            res.append({"prompt_tokens_target": n, "status": None, "error": "could not build a prompt of this length"})
            continue
        r = chat(m, max_tokens=8)
        res.append({"prompt_tokens_target": n, "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"), **brief(r)})
    bad = [x["prompt_tokens_target"] for x in res if not (x.get("status") == 200 and x.get("finish") in ("stop", "length"))]
    return not bad, (f"prompt rows {base}-20: failed at {bad}" if bad else f"prompt rows {base}-20 all complete"), {"runs": res}


def c_system_multi():
    m = [{"role": "system", "content": "You are a terse assistant. Always answer in UPPERCASE letters only."},
         {"role": "user", "content": "Name a colour of the sky."},
         {"role": "assistant", "content": "BLUE"},
         {"role": "user", "content": "Name a colour of grass."}]
    r = chat(m, max_tokens=16)
    c = (r.get("content") or "").strip()
    ok = r["status"] == 200 and "GREEN" in c.upper() and c == c.upper()
    return ok, f"reply {c!r}", brief(r)


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def c_tools():
    # tool_choice "auto": the server refuses "required" for templates without <tool_call> markers (DSML is not one)
    r = chat(user("What is the weather in Lisbon right now? Call the get_weather tool; do not answer from memory."),
             tools=TOOLS, tool_choice="auto", max_tokens=160)
    calls = r.get("tool_calls") or []
    ok = False
    if r["status"] == 200 and r.get("finish") == "tool_calls" and calls:
        try:
            args = json.loads(calls[0]["function"]["arguments"])
            ok = calls[0]["function"]["name"] == "get_weather" and "lisbon" in json.dumps(args).lower()
        except (ValueError, KeyError, TypeError):
            ok = False
    return ok, f"finish {r.get('finish')}, {len(calls)} call(s)", brief(r)


def c_json():
    r = chat(user("Give a JSON object with keys name and age for a fictional person."), response_format={"type": "json_object"}, max_tokens=96)
    try:
        obj = json.loads(r.get("content") or "")
        ok = r["status"] == 200 and isinstance(obj, dict)
    except ValueError:
        ok = False
    return ok, f"reply {(r.get('content') or '')[:120]!r}", brief(r)


def c_thinking():
    on = chat(user("What is 17 * 23? Answer with the number."), max_tokens=512, chat_template_kwargs={"enable_thinking": True})
    off = chat(user("What is 17 * 23? Answer with the number."), max_tokens=64, chat_template_kwargs={"enable_thinking": False})
    ok = on["status"] == 200 and bool(on.get("reasoning")) and "391" in (on.get("content") or "") \
        and off["status"] == 200 and not off.get("reasoning") and "391" in (off.get("content") or "")
    return ok, f"on: reasoning {len(on.get('reasoning') or '')} chars, content {on.get('content')!r}; off: {off.get('content')!r}", \
        {"on": brief(on), "off": brief(off)}


def c_stop():
    r = chat(user("Count from 1 to 10, one number per line."), stop=["5"], max_tokens=64)
    c = r.get("content") or ""
    ok = r["status"] == 200 and r.get("finish") == "stop" and "5" not in c and "4" in c
    return ok, f"content {c!r}", brief(r)


def c_max_tokens():
    r = chat(user("Write a long story about a lighthouse keeper."), max_tokens=5)
    ok = r["status"] == 200 and r.get("finish") == "length" and (r.get("usage") or {}).get("completion_tokens") == 5 and len(r.get("ids") or []) == 5
    return ok, f"finish {r.get('finish')}, {(r.get('usage') or {}).get('completion_tokens')} tokens", brief(r)


def c_capacity():
    m = user("Reply with the single word: ok")
    n = ntok(m)
    if n is None:
        return False, "/v1/tokenize did not count the prompt", {}
    room = A.context - n
    at = chat(m, max_tokens=room, stop=["\n", "."])   # (the stop bounds the reply; the limit check is what is tested)
    over = chat(m, max_tokens=room + 1, stop=["\n", "."])
    err = json.dumps(over.get("error"))
    ok = at["status"] == 200 and over["status"] == 400 and "context" in err.lower()
    return ok, f"prompt {n} tokens: max_tokens {room} -> {at['status']} ({at.get('finish')}), {room + 1} -> {over['status']}", \
        {"at": brief(at), "over": brief(over)}


def c_malformed():
    cases = [("bad json", None, b"{not json"), ("no messages", {"model": A.model}, None),
             ("unknown role", {"model": A.model, "messages": [{"role": "wizard", "content": "x"}]}, None),
             ("negative max_tokens", {"model": A.model, "messages": user("x"), "max_tokens": -3}, None),
             ("image part", {"model": A.model, "messages": [{"role": "user", "content": [
                 {"type": "text", "text": "what is this?"},
                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}]}, None)]
    res = []
    for name, body, raw in cases:
        st, r = request("POST", "/v1/chat/completions", body, raw)
        res.append({"case": name, "status": st, "body": r if isinstance(r, dict) else str(r)[:300]})
    image_text = json.dumps(res[-1]["body"]).lower()
    after = chat(PROBE)
    ok = all(400 <= x["status"] < 500 for x in res) and ("image" in image_text or "text only" in image_text) \
        and after["status"] == 200 and "paris" in (after.get("content") or "").lower()
    return ok, ", ".join(f"{x['case']} {x['status']}" for x in res) + f"; probe after: {after['status']}", {"cases": res, "after": brief(after)}


def c_seeded():
    """Seeded sampling (temperature 0.8, top_k 20, top_p 0.95): one seed twice gives the same token ids, streamed or not;
    another seed gives another reply (the draws are keyed by seed, position and token id)."""
    m = user("Write a short poem about autumn leaves.")
    kw = dict(max_tokens=48, temperature=0.8, top_p=0.95, top_k=20)
    a, b = chat(m, seed=1234, **kw), chat(m, seed=1234, **kw)
    c = chat(m, seed=98765, **kw)
    st = stream(m, seed=1234, **kw)
    same = a.get("ids") is not None and a.get("ids") == b.get("ids")
    differ = c.get("ids") is not None and c.get("ids") != a.get("ids")
    streamed = st["status"] == 200 and not st["error"] and st["text"] == (a.get("content") or "")
    ok = all(r["status"] == 200 for r in (a, b, c)) and same and differ and streamed
    return ok, f"same seed equal {same}, other seed differs {differ}, streamed equal {streamed}", \
        {"a": brief(a), "b": brief(b), "c": brief(c), "stream_text": st["text"][:300]}


def c_concurrent():
    ma, mb = user("Name three rivers in Europe."), user("Name three mountains in Asia.")
    ia, ib = chat(ma, max_tokens=48), chat(mb, max_tokens=48)
    got = {}

    def go(k, m):
        got[k] = chat(m, max_tokens=48)
    ts = [threading.Thread(target=go, args=("a", ma)), threading.Thread(target=go, args=("b", mb))]
    t0 = time.monotonic()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.monotonic() - t0
    ok = all(r["status"] == 200 for r in (ia, ib, got["a"], got["b"])) and got["a"].get("ids") == ia.get("ids") and got["b"].get("ids") == ib.get("ids")
    return ok, f"concurrent pair in {wall:.1f}s; equal to isolated: a {got['a'].get('ids') == ia.get('ids')}, b {got['b'].get('ids') == ib.get('ids')}", \
        {"isolated": [brief(ia), brief(ib)], "concurrent": [brief(got["a"]), brief(got["b"])], "wall_s": wall}


def c_cancel():
    m = user("Write a detailed essay about the history of bridges.")
    iso = chat(m, max_tokens=160)
    s = stream(m, cancel_after=8, max_tokens=160)
    time.sleep(1.0)
    again = chat(m, max_tokens=160)
    probe = chat(PROBE)
    ok = s["cancelled"] and iso["status"] == 200 and again["status"] == 200 and again.get("ids") == iso.get("ids") \
        and "paris" in (probe.get("content") or "").lower()
    return ok, f"dropped after {s['events']} events; rerun equal to isolated: {again.get('ids') == iso.get('ids')}; probe {probe['status']}", \
        {"isolated": brief(iso), "dropped": {k: s[k] for k in ("status", "events", "cancelled", "error")}, "again": brief(again), "probe": brief(probe)}


def filler(n_tokens, system=True):
    """A conversation whose rendered prompt has exactly n_tokens tokens (by /v1/tokenize), or None."""
    words = "the river runs past the old mill and under the stone bridge toward the sea".split()
    tail = "\nHow many times does the word river appear? Answer with a number." if n_tokens > 40 else ""
    k = max(0, n_tokens - (ntok(user("x" + tail, system)) or 0))
    for _ in range(8):
        text = " ".join(words[i % len(words)] for i in range(k)) if k else ""
        m = user((text or "x") + tail, system)
        n = ntok(m)
        if n is None:
            return None
        if n == n_tokens:
            return m
        k += n_tokens - n
        if k < 0:
            return None
    return None


def c_long_prompt():
    """Prompts across the 128-row and 2,048-row chunk boundaries, 4K, 8K and (with a 16K pool) 16K rows, then one that
    fills the pool to 32 positions under capacity (every KV position written; the exact-capacity limit itself is the
    capacity case's)."""
    if not A.long:
        return None, "skipped: prompts over 128 rows are not enabled for this run (--long 0)", {}
    res = []
    sizes = [127, 128, 129, 2047, 2048, 2049, 4096, 8000] + ([16000] if A.context >= 16384 else [])
    for n in sizes:
        m = filler(n)
        if m is None:
            res.append({"prompt_tokens_target": n, "error": "could not build a prompt of this length"})
            continue
        r = chat(m, max_tokens=16)
        res.append({"prompt_tokens_target": n, **brief(r), "prefill_seconds": r.get("prefill_seconds")})
    m = filler(A.context - 32)
    at = chat(m, max_tokens=16, ignore_eos=True) if m else {"status": None, "error": "no prompt"}
    res.append({"near_capacity_fill": A.context - 16, **brief(at), "prefill_seconds": at.get("prefill_seconds")})
    ok = all(x.get("status") == 200 and x.get("finish") in ("stop", "length") for x in res)
    return ok, "; ".join(f"{x.get('prompt_tokens_target', x.get('near_capacity_fill'))}: {x.get('status')} {x.get('finish')}" for x in res), {"runs": res}


REPLAY_B = "Summarise the passage above in one sentence, then name the season it describes."


def replay_prompts():
    """A: 2,049 rows (a full 2,048-row chunk and a 1-row final chunk, the replay floor's small-final-chunk case); B: an
    unrelated 2,050-row prompt (another word set and question)."""
    a = filler(2049)
    words = "snow falls on the quiet orchard while lanterns glow in the farmhouse windows".split()
    base = ntok(user("x\n" + REPLAY_B)) or 0
    k, b = max(1, 2050 - base), None
    for _ in range(8):
        m = user(" ".join(words[i % len(words)] for i in range(k)) + "\n" + REPLAY_B)
        n = ntok(m)
        if n == 2050:
            b = m
            break
        k += 2050 - (n or 0)
    return a, b


def c_replay_boundary():
    """Consecutive unrelated requests across the replay boundary: A (a 1-row final chunk) then B; B again on the same
    server must equal itself, and replay_fresh (the restart's suite) compares B with a fresh server's first request."""
    if not A.long:
        return None, "skipped: needs prompts over 128 rows (--long 0)", {}
    a, b = replay_prompts()
    if a is None or b is None:
        return False, "could not build the replay-boundary prompts", {}
    ra = chat(a, max_tokens=16)
    rb = chat(b, max_tokens=32)
    rb2 = chat(b, max_tokens=32)
    ok = all(r["status"] == 200 for r in (ra, rb, rb2)) and rb.get("ids") == rb2.get("ids") and bool(rb.get("ids"))
    return ok, f"A {ra['status']} ({(ra.get('usage') or {}).get('prompt_tokens')} rows), B after A {rb['status']}, B again equal {rb.get('ids') == rb2.get('ids')}", \
        {"a": brief(ra), "b_after_a": brief(rb), "b_again": brief(rb2), "b_ids": rb.get("ids")}


def c_replay_fresh():
    """(restart suite, --reference = the first start's serve_qual JSON) B as a fresh server's first request: the same
    token ids as B after A on the first start (no state left by A changes B)."""
    if not A.long:
        return None, "skipped: needs prompts over 128 rows (--long 0)", {}
    if not A.reference:
        return None, "skipped: no --reference (the first start's serve_qual JSON)", {}
    ref = next((c for c in json.load(open(A.reference))["cases"] if c["case"] == "replay_boundary"), None)
    if not ref or ref["status"] != "pass":
        return None, "skipped: the first start's replay_boundary case did not pass", {}
    _, b = replay_prompts()
    rb = chat(b, max_tokens=32)
    ok = rb["status"] == 200 and rb.get("ids") == ref["data"]["b_ids"]
    return ok, f"fresh B equal to B after A: {rb.get('ids') == ref['data']['b_ids']}", {"fresh_b": brief(rb)}


def c_drafts():
    if not A.drafts:
        return None, "skipped: DSpark is off in this configuration (--drafts 0)", {}
    res = []
    for text in ("Write a Python function that reverses a linked list.", "Explain photosynthesis in three sentences.",
                 "Count from 1 to 40 separated by spaces."):
        plain = chat(user(text), max_tokens=128, draft=False)
        spec = chat(user(text), max_tokens=128, draft=True)
        sp = spec.get("speculative") or {}
        res.append({"prompt": text, "equal": plain.get("ids") == spec.get("ids"), "plain": brief(plain), "spec": brief(spec),
                    "drafted": sp.get("drafted"), "accepted": sp.get("accepted"), "rounds": sp.get("rounds")})
    # seeded sampling: a verify row draws as the serial run at its position, so DSpark must not change the reply
    kw = dict(max_tokens=96, temperature=0.8, top_p=0.95, top_k=20, seed=4242)
    plain = chat(user("Describe a busy harbour at dawn."), draft=False, **kw)
    spec = chat(user("Describe a busy harbour at dawn."), draft=True, **kw)
    sp = spec.get("speculative") or {}
    res.append({"prompt": "seeded harbour (t 0.8, seed 4242)", "equal": plain.get("ids") == spec.get("ids"), "plain": brief(plain),
                "spec": brief(spec), "drafted": sp.get("drafted"), "accepted": sp.get("accepted"), "rounds": sp.get("rounds")})
    ok = all(x["equal"] and (x["drafted"] or 0) > 0 for x in res) and any((x["accepted"] or 0) > 0 for x in res)
    return ok, "; ".join(f"equal {x['equal']} drafted {x['drafted']} accepted {x['accepted']}" for x in res), {"runs": res}


def spec_ok(r):
    """DSpark accounting of one reply: committed ids = usage tokens; accepted <= drafted; rounds <= tokens; tokens/round."""
    sp = r.get("speculative") or {}
    ids, ct = r.get("ids") or [], (r.get("usage") or {}).get("completion_tokens")
    rounds, dr, ac = sp.get("rounds") or 0, sp.get("drafted") or 0, sp.get("accepted") or 0
    ok = r["status"] == 200 and len(ids) == ct and 0 <= ac <= dr and rounds <= max(ct or 0, 1) \
        and (rounds == 0 or abs((sp.get("tokens_per_round") or 0) - ct / rounds) < 1e-6)
    return ok, {"tokens": ct, "rounds": rounds, "drafted": dr, "accepted": ac}


def c_drafts_budget():
    """Output-budget truncation under DSpark: a round can commit several tokens, but max_tokens 1, 2, 3, 5, 7 must give
    exactly that many (finish length), the same ids as without drafts."""
    if not A.drafts:
        return None, "skipped: DSpark is off in this configuration (--drafts 0)", {}
    m = user("Count from 1 to 60, separated by commas, nothing else.")
    res = []
    for mt in (1, 2, 3, 5, 7):
        a, b = chat(m, max_tokens=mt, draft=True), chat(m, max_tokens=mt, draft=False)
        acc_ok, acc = spec_ok(a)
        res.append({"max_tokens": mt, "tokens": (a.get("usage") or {}).get("completion_tokens"), "finish": a.get("finish"),
                    "equal": a.get("ids") == b.get("ids"), "accounting": acc, "accounting_ok": acc_ok})
    ok = all(x["tokens"] == x["max_tokens"] and x["finish"] == "length" and x["equal"] and x["accounting_ok"] for x in res)
    return ok, "; ".join(f"{x['max_tokens']}: {x['tokens']} {x['finish']} equal {x['equal']}" for x in res), {"runs": res}


def c_drafts_eos():
    """EOS inside a proposal: short answers with drafts end with finish stop and the ids of the run without drafts."""
    if not A.drafts:
        return None, "skipped: DSpark is off in this configuration (--drafts 0)", {}
    res = []
    for text in ("Reply with the single word OK.", "What is 2 + 2? Answer with the digit only.", "Say yes or no: is water wet?"):
        a, b = chat(user(text), max_tokens=32, draft=True), chat(user(text), max_tokens=32, draft=False)
        acc_ok, acc = spec_ok(a)
        res.append({"prompt": text, "finish": a.get("finish"), "equal": a.get("ids") == b.get("ids"), "accounting": acc, "accounting_ok": acc_ok,
                    "content": a.get("content")})
    ok = all(x["finish"] == "stop" and x["equal"] and x["accounting_ok"] for x in res)
    return ok, "; ".join(f"{x['content']!r} {x['finish']} equal {x['equal']}" for x in res), {"runs": res}


def c_drafts_cancel():
    """Cancellation with DSpark: a stream dropped after 8 events, then the same request: identical to the isolated reply
    with and without drafts; the server keeps serving."""
    if not A.drafts:
        return None, "skipped: DSpark is off in this configuration (--drafts 0)", {}
    m = user("Write a detailed essay about the history of bridges.")
    iso = chat(m, max_tokens=160, draft=True)
    s = stream(m, cancel_after=8, max_tokens=160, draft=True)
    time.sleep(1.0)
    again = chat(m, max_tokens=160, draft=True)
    plain = chat(m, max_tokens=160, draft=False)
    ok = s["cancelled"] and again.get("ids") == iso.get("ids") == plain.get("ids") and again["status"] == 200
    return ok, f"dropped after {s['events']} events; rerun equal {again.get('ids') == iso.get('ids')}, equal to no drafts {iso.get('ids') == plain.get('ids')}", \
        {"iso": brief(iso), "again": brief(again), "plain": brief(plain)}


def c_drafts_accounting():
    """DSpark accounting on varied replies: committed ids = usage tokens, accepted <= drafted, tokens per round = tokens /
    rounds; proposals are made and some accepted (a predictable reply) and acceptance stays partial on prose."""
    if not A.drafts:
        return None, "skipped: DSpark is off in this configuration (--drafts 0)", {}
    res = []
    for text in ("Count from 1 to 60, separated by commas, nothing else.", "Write a short poem about the sea in free verse.",
                 "Write a Python function that returns the n-th Fibonacci number, with a docstring."):
        r = chat(user(text), max_tokens=120, draft=True)
        acc_ok, acc = spec_ok(r)
        res.append({"prompt": text, **acc, "accounting_ok": acc_ok})
    ok = all(x["accounting_ok"] and x["drafted"] > 0 for x in res) and any(x["accepted"] > 0 for x in res)
    return ok, "; ".join(f"{x['tokens']} tokens / {x['rounds']} rounds, accepted {x['accepted']} of {x['drafted']}" for x in res), {"runs": res}


CASES = {name[2:]: fn for name, fn in globals().items() if name.startswith("c_")}


def main():
    global A
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("out")
    p.add_argument("--model", default="dsv41")
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--long", type=int, default=0)
    p.add_argument("--drafts", type=int, default=0)
    # (replay_boundary / replay_fresh belong to plan workstream A1: run them by name; they stay out of the default
    # pass/fail counts until A1's rules are final)
    p.add_argument("--cases", default=",".join(c for c in CASES if c not in ("replay_boundary", "replay_fresh")))
    p.add_argument("--reference", default=None, help="the first start's serve_qual JSON (replay_fresh)")
    A = p.parse_args()
    results = []
    for name in A.cases.split(","):
        t0 = time.monotonic()
        try:
            ok, detail, data = CASES[name]()
        except Exception as e:  # a case that cannot complete fails; it never counts as passed
            ok, detail, data = False, f"exception {type(e).__name__}: {e}", {}
        rec = {"case": name, "ok": ok, "status": "pass" if ok else ("skipped" if ok is None else "fail"), "detail": detail,
               "seconds": round(time.monotonic() - t0, 2), "data": data}
        results.append(rec)
        print(f"{rec['status']:7s} {name}: {detail}", flush=True)
    summary = {s: sum(1 for r in results if r["status"] == s) for s in ("pass", "fail", "skipped")}
    doc = {"schema": "tf-port-serve-qual/1", "url": A.url, "model": A.model, "context": A.context, "long": A.long,
           "drafts": A.drafts, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "summary": summary, "cases": results}
    doc["sha256"] = hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()
    json.dump(doc, open(A.out, "w"), indent=1)
    print(json.dumps(summary))
    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
