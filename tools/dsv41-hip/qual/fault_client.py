#!/usr/bin/env python3
"""fault_client.py URL OUT.jsonl [--model NAME] [--max-tokens N] [--prompt-words N] [--timeout S]

The client side of a served fault case (served_faults.sh): one streamed greedy request whose every token-bearing event
is written to OUT.jsonl as it arrives ({"t": seconds since the request, "w": wall clock, "n": events so far}), so the driver can inject
its fault mid-decode (or mid-prefill with --prompt-words) and later compare event times with the fault's time. The
last line is the outcome: {"end": "done" | "error_event" | "closed" | "http_error" | "timeout", "finish": ..., "events": N,
"error": ..., "t_end": ...}. "closed" means the connection ended without a finish reason or [DONE].
Standard library only.
"""
import argparse
import http.client
import json
import socket
import time
import urllib.parse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("out")
    p.add_argument("--model", default="dsv41")
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--prompt-words", type=int, default=0, help="a filler prompt of about this many words (prefill faults)")
    p.add_argument("--timeout", type=float, default=120)
    a = p.parse_args()
    text = "Write a very long, detailed history of bridge engineering from antiquity to today."
    if a.prompt_words:
        words = "the river runs past the old mill and under the stone bridge toward the sea".split()
        text = " ".join(words[i % len(words)] for i in range(a.prompt_words)) + "\n" + text
    body = {"model": a.model, "messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": a.max_tokens,
            "stream": True, "ignore_eos": True}
    u = urllib.parse.urlparse(a.url)
    out = open(a.out, "w", buffering=1)
    t0 = time.monotonic()
    n, finish, err, end = 0, None, None, None
    try:
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=a.timeout)
        c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        out.write(json.dumps({"t": round(time.monotonic() - t0, 3), "http": r.status}) + "\n")
        if r.status != 200:
            end, err = "http_error", r.read().decode("utf-8", "replace")[:2000]
        else:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    end = end or "done"
                    break
                obj = json.loads(data)
                if obj.get("error"):
                    end, err = "error_event", obj["error"]
                for ch in obj.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("content") or d.get("reasoning_content"):
                        n += 1
                        out.write(json.dumps({"t": round(time.monotonic() - t0, 3), "w": round(time.time(), 3), "n": n}) + "\n")
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
            end = end or ("done" if finish else "closed")
    except socket.timeout as e:
        end, err = "timeout", str(e)
    except (http.client.HTTPException, ConnectionError, OSError) as e:
        end, err = "closed", f"{type(e).__name__}: {e}"
    out.write(json.dumps({"end": end, "finish": finish, "events": n, "error": err, "t_end": round(time.monotonic() - t0, 3),
                         "w_end": round(time.time(), 3)}) + "\n")


if __name__ == "__main__":
    main()
