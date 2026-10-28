"""bench_stream.py URL NAME CASES REPEAT MAX_TOKENS TIMEOUT OUT: the server's timed requests (serve-rank0.sh bench).

CASES is a comma list of code, prose (the decode cases: fixed prompts of about 30 tokens, the ones first light was
measured with) and prefill (a sentence REPEAT times, then a question). Each is one streamed greedy chat request of
MAX_TOKENS tokens, run one after another. Per case: prompt and completion tokens (the server's usage), the content
events, TTFT (to the first content event), decode tok/s = (completion tokens - 1) / (last event - first event), the
same rate counted in events (a token whose text is empty or split sends none), prefill tok/s = prompt tokens / TTFT.
A failed case is recorded with its error and the others still run. OUT gets every case with its reply and the server's
/metrics after the last; stdout one JSON line without them."""
import json
import sys
import time
import urllib.request

PROMPTS = {
    "code": "Write a Python class implementing an LRU cache with get/put in O(1), with type hints and a short usage example.",
    "prose": "Explain to a new manager, in about 300 words, why code review slows teams down in the short term but speeds them up over a year.",
}


def prefill_prompt(repeat):
    return "The quick brown fox jumps over the lazy dog. " * repeat + "How many times did the fox jump? Answer in one short sentence."


def run(url, name, prompt, max_tokens, timeout):
    body = json.dumps({"model": name, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    events, text, usage = 0, [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta", {})
                piece = d.get("content") or d.get("reasoning_content")
                if piece:
                    now = time.monotonic() - t0
                    first = now if first is None else first
                    last = now
                    events += 1
                    text.append(piece)
    prompt_tokens = (usage or {}).get("prompt_tokens")
    done = (usage or {}).get("completion_tokens") or events
    span = (last - first) if first is not None and last > first else None
    return {"prompt_tokens": prompt_tokens, "completion_tokens": done, "events": events,
            "ttft_s": round(first, 4) if first is not None else None, "total_s": round(time.monotonic() - t0, 3),
            "decode_tok_per_s": round((done - 1) / span, 2) if span and done > 1 else None,
            "decode_events_per_s": round((events - 1) / span, 2) if span and events > 1 else None,
            "prefill_tok_per_s": round(prompt_tokens / first, 1) if prompt_tokens and first else None,
            "reply": "".join(text), "usage": usage}


def main():
    url, name, cases, repeat, max_tokens, timeout, out = sys.argv[1:8]
    recs = []
    for case in cases.split(","):
        prompt = prefill_prompt(int(repeat)) if case == "prefill" else PROMPTS[case]
        try:
            rec = run(url, name, prompt, int(max_tokens), float(timeout))
        except Exception as e:  # recorded; the next case still runs
            rec = {"error": f"{type(e).__name__}: {e}"}
        recs.append({"case": case, **rec})
    summary = {"cases": [{k: v for k, v in r.items() if k not in ("reply", "usage")} for r in recs],
               "uptime": open("/proc/loadavg").read().strip()}
    try:
        metrics = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    except Exception as e:
        metrics = f"unavailable: {e}"
    json.dump({**summary, "cases_full": recs, "metrics": metrics}, open(out, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
