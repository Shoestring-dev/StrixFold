# l-link: prompt prefill timing on the served engine (rank 0's /v1/chat/completions), the lead's long_prompt.py
# prompts (2k, 4k) twice each; each measure = one max_tokens=1 request (prefill wall) and one streamed request with a
# different trailing question (TTFT and decode)
import json, sys, time, urllib.request
URL = "http://127.0.0.1:18080/v1/chat/completions"
src = open("~/src/ds4/tests/test-vectors/flash-vision-exp/prompts/long_memory_archive.txt").read()
def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)
def measure(name, doc, q, gen):
    msgs = [{"role": "user", "content": doc + "\n\nQuestion: " + q}]
    t0 = time.monotonic(); d = json.load(post({"model": "dsv41", "messages": msgs, "temperature": 0, "max_tokens": 1})); t1 = time.monotonic()
    pt = d["usage"]["prompt_tokens"]
    body = {"model": "dsv41", "messages": [{"role": "user", "content": doc + "\n\nQuestion (answer in detail): " + q}],
            "temperature": 0, "max_tokens": gen, "stream": True}
    t2 = time.monotonic(); first = last = None; n = 0; text = []
    with post(body) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            ch = json.loads(line[5:]).get("choices") or []
            c = ch[0].get("delta", {}).get("content") if ch else None
            if c:
                now = time.monotonic(); n += 1; first = first or now; last = now; text.append(c)
    print(json.dumps({"name": name, "prompt_tokens": pt, "prefill_wall_s": round(t1 - t0, 3), "prefill_tok_per_s": round(pt / (t1 - t0), 1),
                      "ttft_s": round(first - t2, 3), "prefill_tok_per_s_stream": round(pt / (first - t2), 1),
                      "decode_events": n, "decode_tok_per_s": round((n - 1) / (last - first), 2) if n > 1 else 0,
                      "answer": "".join(text)}), flush=True)
sizes = [(22000, "4k"), (45000, "8k")]
for rep in range(int(sys.argv[1]) if len(sys.argv) > 1 else 2):
    for chars, name in sizes:
        measure(f"{name}#{rep}", ((src + "\n") * 3)[:chars], "Summarise the main events described above in a few sentences.", 200)
