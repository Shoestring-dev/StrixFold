"""hc_session.py URL OUT: a long session against one server (TF_HC_FUSE=1) for _hc_fused's arrival counters: the code /
prose cases first, then long greedy requests until ~1,100 rounds, an early stop (max_tokens 7), a request abandoned
mid-stream (the connection closed after 40 events: the server cancels it between rounds), a stop sequence, then the code
/ prose cases again. A stale counter would leave a round's Sinkhorn unrun or run early, so the replies would change:
rank 0's done lines (rounds, sha) are compared afterwards."""
import json, socket, sys, time, urllib.request
url, out = sys.argv[1], sys.argv[2]
CASES = {
    "code": "Write a Python class implementing an LRU cache with get/put in O(1), with type hints and a short usage example.",
    "prose": "Explain to a new manager, in about 300 words, why code review slows teams down in the short term but speeds them up over a year.",
}
LONG = [
    "Write a detailed history of the printing press and its effect on European science, in plain prose.",
    "Describe, step by step, how a compiler turns source code into machine code, with examples in C.",
    "Write a long story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how TCP congestion control works, covering slow start, AIMD and fast recovery, with numbers.",
    "Write a beginner's guide to baking sourdough bread, from starter to the first loaf.",
    "Explain the causes and consequences of the 2008 financial crisis to a high-school student.",
    "Write Rust code for a thread-safe bounded queue with tests, and explain each part.",
    "Compare three sorting algorithms in depth, with their complexity, stability and when to use each.",
]


def req(prompt, max_tokens, stop=None, abandon_after=None):
    body = {"model": "dsv41", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if stop:
        body["stop"] = stop
    r = urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=600)
    events, text = 0, []
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        for ch in json.loads(data).get("choices") or []:
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                events += 1
                text.append(piece)
        if abandon_after and events >= abandon_after:
            r.close()
            return {"abandoned_after_events": events}
    return {"events": events, "chars": len("".join(text))}


log = []
def step(name, *a, **kw):
    t0 = time.monotonic()
    try:
        res = req(*a, **kw)
    except Exception as e:  # an abandoned stream may raise on close
        res = {"error": type(e).__name__}
    res.update(name=name, s=round(time.monotonic() - t0, 2))
    log.append(res)
    print(json.dumps(res), flush=True)
    time.sleep(0.5)

for k, p in CASES.items():
    step("pre-" + k, p, 400)
for i, p in enumerate(LONG * 3):
    step(f"long-{i}", p, 400)
    if i == 3:
        step("early-stop", LONG[0], 7)
        step("abandon", LONG[1], 400, abandon_after=40)
        step("stop-seq", LONG[2], 400, stop=["the"])
    if i >= 7 and sys.argv[3:] and sum(1 for x in log if x["name"].startswith("long")) * float(sys.argv[3]) >= 1100:
        break
for k, p in CASES.items():
    step("post-" + k, p, 400)
json.dump(log, open(out, "w"), indent=1)
