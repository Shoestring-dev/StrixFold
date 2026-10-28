#!/usr/bin/env python3
"""struct_serve.py URL OUT.json [--model NAME] [--drafts 0|1] [--cases a,b,...] [--reference REF.json]

Structured output on the real server (handoff section 2's first-release scope: structured JSON, tools; qualification
items U44, U45). Standard library only; runs on rank 0's box against 127.0.0.1, like serve_qual.py (its helpers).

Cases (each ok true / false, with the replies as evidence):
  json_object     response_format json_object on three prompts: each reply is one JSON object (json.loads); every
                  JSON reply also starts at its value and has no blank line or carriage return
  json_schema     response_format json_schema, three schemas (flat with bounds and an enum; nested arrays of objects;
                  $defs with anyOf and null): each reply parses and validates (validator below: type, properties,
                  required, additionalProperties false, items, enum, const, min/max Length/Items, minimum/maximum,
                  anyOf, $ref to #/$defs)
  json_stream     the flat schema streamed: the assembled content validates
  json_thinking   the nested schema with enable_thinking: reasoning present, the content after it validates
  choice          guided_choice: the reply is one of the choices
  regex           guided_regex: the reply matches the pattern in full
  tools_required  tool_choice "required" with two tools on a prompt that needs neither: finish_reason tool_calls, each
                  call to an offered tool with arguments that parse and carry its required parameters
  tools_named     tool_choice naming one tool: that tool is called, its arguments valid
  refusals        a schema keyword the grammar cannot enforce, a bad regex, response_format beside a required call,
                  guided_grammar: 400 each with an error message
  plain           three greedy prompts with no structure: token ids recorded; with --reference (the same prompts on
                  the build without this change), they must be identical
  drafts          (--drafts 1) the flat schema with draft true and false: both valid, drafts proposed under the grammar
"""
import argparse
import http.client
import json
import re
import sys
import time
import urllib.parse

A = None
TIMEOUT = 900


def conn():
    u = urllib.parse.urlparse(A.url)
    return http.client.HTTPConnection(u.hostname, u.port, timeout=TIMEOUT)


def request(method, path, body):
    c = conn()
    c.request(method, path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    text = r.read().decode("utf-8", "replace")
    c.close()
    try:
        return r.status, json.loads(text)
    except ValueError:
        return r.status, text


def user(text):
    return [{"role": "user", "content": text}]


def chat(messages, **kw):
    b = {"model": A.model, "messages": messages, "temperature": 0, "max_tokens": 400, "return_token_ids": True}
    b.update(kw)
    t0 = time.monotonic()
    st, r = request("POST", "/v1/chat/completions", b)
    wall = time.monotonic() - t0
    if st != 200 or not isinstance(r, dict):
        return {"status": st, "error": r, "wall_s": round(wall, 3)}
    ch = r["choices"][0]
    tf = r.get("tensorfold") or {}
    usage = r.get("usage") or {}
    return {"status": st, "wall_s": round(wall, 3), "finish": ch.get("finish_reason"), "content": ch["message"].get("content"),
            "reasoning": ch["message"].get("reasoning_content"), "tool_calls": ch["message"].get("tool_calls"),
            "completion_tokens": usage.get("completion_tokens"), "ids": tf.get("token_ids"), "speculative": r.get("speculative"),
            "tok_s": round(usage.get("completion_tokens", 0) / wall, 2) if wall > 0 else None}


def stream(messages, **kw):
    b = {"model": A.model, "messages": messages, "temperature": 0, "max_tokens": 400, "stream": True}
    b.update(kw)
    c = conn()
    c.request("POST", "/v1/chat/completions", body=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    out = {"status": r.status, "text": [], "finish": None, "error": None}
    if r.status != 200:
        out["error"] = r.read().decode("utf-8", "replace")
        c.close()
        return out
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
        for ch in obj.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content"):
                out["text"].append(d["content"])
            if ch.get("finish_reason"):
                out["finish"] = ch["finish_reason"]
    c.close()
    out["text"] = "".join(out["text"])
    return out


def brief(r):
    return {k: r.get(k) for k in ("status", "finish", "content", "reasoning", "tool_calls", "completion_tokens", "tok_s", "wall_s", "error", "speculative")}


# -- a JSON schema check for the subset the cases use ----------------------------------------------------------------

def valid(schema, x, root=None):
    root = root if root is not None else schema
    if not isinstance(schema, dict):
        return True
    if "$ref" in schema:
        ref = schema["$ref"]
        target = root
        for part in ref.lstrip("#").strip("/").split("/"):
            if part:
                target = target[part]
        return valid(target, x, root)
    if "anyOf" in schema:
        return any(valid(s, x, root) for s in schema["anyOf"])
    if "enum" in schema:
        return x in schema["enum"]
    if "const" in schema:
        return x == schema["const"]
    t = schema.get("type")
    if t is not None:
        ok = False
        for n in (t if isinstance(t, list) else [t]):
            ok |= {"object": isinstance(x, dict), "array": isinstance(x, list), "string": isinstance(x, str),
                   "boolean": isinstance(x, bool), "null": x is None,
                   "integer": isinstance(x, int) and not isinstance(x, bool),
                   "number": isinstance(x, (int, float)) and not isinstance(x, bool)}[n]
        if not ok:
            return False
    if isinstance(x, dict):
        props = schema.get("properties")
        if any(r not in x for r in schema.get("required", [])):
            return False
        for k, v in x.items():
            if props is not None and k not in props:
                return False
            if props is not None and not valid(props[k], v, root):
                return False
    if isinstance(x, list):
        if len(x) < schema.get("minItems", 0) or len(x) > schema.get("maxItems", 1 << 60):
            return False
        if "items" in schema and not all(valid(schema["items"], v, root) for v in x):
            return False
    if isinstance(x, str) and not (schema.get("minLength", 0) <= len(x) <= schema.get("maxLength", 1 << 60)):
        return False
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if x < schema.get("minimum", -1e300) or x > schema.get("maximum", 1e300):
            return False
    return True


def parses(text, schema=None):
    try:
        v = json.loads(text)
    except (TypeError, ValueError) as e:
        return False, f"not JSON: {e}"
    # the grammar's whitespace rule: nothing before the value, at most one newline a run, no carriage return
    if text[:1].isspace() or re.search(r"\n[ \t]*\n", text) or "\r" in text:
        return False, f"untidy whitespace: {text[:80]!r}"
    if schema is None:
        return isinstance(v, dict), "an object" if isinstance(v, dict) else "not an object"
    return (True, "valid") if valid(schema, v) else (False, "does not match the schema")


PERSON = {"type": "object", "properties": {
    "name": {"type": "string", "minLength": 1, "maxLength": 40},
    "age": {"type": "integer", "minimum": 0, "maximum": 120},
    "occupation": {"type": "string", "enum": ["engineer", "teacher", "artist", "doctor", "farmer"]},
    "email": {"type": ["string", "null"]},
    "hobbies": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
    "required": ["name", "age", "occupation", "hobbies"], "additionalProperties": False}
FORECAST = {"type": "object", "properties": {
    "city": {"type": "string"},
    "days": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "object", "properties": {
        "day": {"type": "string", "enum": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]},
        "high_c": {"type": "number", "minimum": -60, "maximum": 60},
        "rain": {"type": "boolean"}}, "required": ["day", "high_c", "rain"], "additionalProperties": False}}},
    "required": ["city", "days"], "additionalProperties": False}
SHOPPING = {"$defs": {"item": {"type": "object", "properties": {
    "name": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}, "note": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
    "required": ["name", "qty", "note"], "additionalProperties": False}},
    "type": "object", "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/item"}, "minItems": 1, "maxItems": 6}},
    "required": ["items"], "additionalProperties": False}
SCHEMAS = [("person", PERSON, "Invent a fictional person and describe them."),
           ("forecast", FORECAST, "Make up a short weather forecast for Lisbon."),
           ("shopping", SHOPPING, "Write a shopping list for pancakes.")]


def rf(name, schema):
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
         {"type": "function", "function": {"name": "search_web", "description": "Search the web.", "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["query"]}}}]


def calls_ok(r, allowed):
    calls = r.get("tool_calls") or []
    if r.get("finish") != "tool_calls" or not calls:
        return False, f"finish {r.get('finish')}, {len(calls)} calls, content {r.get('content')!r}"
    by = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
    for c in calls:
        f = c["function"]
        if f["name"] not in allowed:
            return False, f"called {f['name']}"
        try:
            args = json.loads(f["arguments"])
        except ValueError as e:
            return False, f"arguments not JSON: {e}"
        if not valid(by[f["name"]], args):
            return False, f"arguments {args} do not match {f['name']}'s parameters"
    return True, "; ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in calls)


def c_json_object():
    res = []
    for p in ("Give a JSON object with keys name and age for a fictional person.", "List three colors.", "What is the capital of France?"):
        r = chat(user(p), response_format={"type": "json_object"})
        ok, why = parses(r.get("content"))
        res.append({"prompt": p, "ok": ok and r["status"] == 200 and r.get("finish") == "stop", "why": why, "reply": brief(r)})
    return all(x["ok"] for x in res), "; ".join(f"{x['why']} ({x['reply']['finish']})" for x in res), {"runs": res}


def c_json_schema():
    res = []
    for name, schema, p in SCHEMAS:
        r = chat(user(p), response_format=rf(name, schema))
        ok, why = parses(r.get("content"), schema)
        res.append({"schema": name, "ok": ok and r.get("finish") == "stop", "why": why, "reply": brief(r)})
    return all(x["ok"] for x in res), "; ".join(f"{x['schema']}: {x['why']}" for x in res), {"runs": res}


def c_json_stream():
    s = stream(user(SCHEMAS[0][2]), response_format=rf("person", PERSON))
    ok, why = parses(s["text"], PERSON)
    return ok and s["status"] == 200 and s["finish"] == "stop", f"{why}, finish {s['finish']}", s


def c_json_thinking():
    r = chat(user(SCHEMAS[1][2]), response_format=rf("forecast", FORECAST), max_tokens=1500, chat_template_kwargs={"enable_thinking": True})
    ok, why = parses(r.get("content"), FORECAST)
    return ok and bool(r.get("reasoning")) and r.get("finish") == "stop", f"reasoning {len(r.get('reasoning') or '')} chars; content {why}", brief(r)


def c_choice():
    choices = ["positive", "negative", "neutral"]
    r = chat(user("What is the sentiment of: 'I absolutely love this new phone!'"), guided_choice=choices, max_tokens=16)
    return r.get("content") in choices and r.get("finish") == "stop", f"content {r.get('content')!r}", brief(r)


def c_regex():
    pattern = r"\(\d{3}\) \d{3}-\d{4}"
    r = chat(user("Make up a US phone number."), guided_regex=pattern, max_tokens=32)
    c = r.get("content") or ""
    return re.fullmatch(pattern, c) is not None and r.get("finish") == "stop", f"content {c!r}", brief(r)


def c_tools_required():
    r = chat(user("Hi there! How are you today?"), tools=TOOLS, tool_choice="required")
    ok, why = calls_ok(r, {"get_weather", "search_web"})
    return ok, why, brief(r)


def c_tools_named():
    r = chat(user("I want to know about the history of Lisbon."), tools=TOOLS, tool_choice={"type": "function", "function": {"name": "get_weather"}})
    ok, why = calls_ok(r, {"get_weather"})
    return ok, why, brief(r)


def c_refusals():
    m = user("x")
    cases = [("uniqueItems", {"response_format": rf("u", {"type": "array", "uniqueItems": True})}),
             ("bad regex", {"guided_regex": "(a)\\1"}),
             ("structure beside a required call", {"response_format": {"type": "json_object"}, "tools": TOOLS, "tool_choice": "required"}),
             ("guided_grammar", {"guided_grammar": "root ::= \"a\""})]
    res = []
    for name, kw in cases:
        st, body = request("POST", "/v1/chat/completions", {"model": A.model, "messages": m, "max_tokens": 4, **kw})
        msg = body.get("error", {}).get("message") if isinstance(body, dict) else body
        res.append({"case": name, "status": st, "message": msg})
    ok = all(x["status"] == 400 and x["message"] for x in res)
    return ok, "; ".join(f"{x['case']}: {x['status']} {x['message']!r}" for x in res), {"runs": res}


PLAIN = ["Answer in one sentence: what is the capital of France?", "Write a haiku about the sea.", "List the first five prime numbers."]


def c_plain():
    res = [{"prompt": p, "reply": brief(r), "ids": r.get("ids")} for p in PLAIN for r in [chat(user(p), max_tokens=64)]]
    if not A.reference:
        return all(x["reply"]["status"] == 200 for x in res), "recorded (no reference given)", {"runs": res}
    ref = {x["prompt"]: x["ids"] for x in json.load(open(A.reference))["cases"]["plain"]["data"]["runs"]}
    same = [x["ids"] == ref.get(x["prompt"]) and x["ids"] is not None for x in res]
    return all(same), f"identical token ids to the reference: {same}", {"runs": res, "reference": A.reference}


def c_drafts():
    if not A.drafts:
        return None, "skipped: DSpark is off in this configuration (--drafts 0)", {}
    res = []
    for d in (False, True):
        r = chat(user(SCHEMAS[0][2]), response_format=rf("person", PERSON), draft=d)
        ok, why = parses(r.get("content"), PERSON)
        res.append({"draft": d, "ok": ok, "why": why, "reply": brief(r)})
    sp = res[1]["reply"].get("speculative") or {}
    ok = all(x["ok"] for x in res) and (sp.get("drafted") or 0) > 0
    return ok, f"{[x['why'] for x in res]}, drafted {sp.get('drafted')} accepted {sp.get('accepted')}", {"runs": res}


CASES = {name[2:]: fn for name, fn in globals().items() if name.startswith("c_")}


def main():
    global A
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("out")
    p.add_argument("--model", default="dsv41")
    p.add_argument("--drafts", type=int, default=0)
    p.add_argument("--cases", default=",".join(CASES))
    p.add_argument("--reference", default=None)
    A = p.parse_args()
    doc = {"url": A.url, "model": A.model, "drafts": A.drafts, "cases": {}}
    for name in A.cases.split(","):
        t0 = time.monotonic()
        try:
            ok, detail, data = CASES[name]()
        except Exception as e:  # noqa: BLE001
            ok, detail, data = False, f"raised {type(e).__name__}: {e}", {}
        doc["cases"][name] = {"ok": ok, "detail": detail, "seconds": round(time.monotonic() - t0, 1), "data": data}
        print(json.dumps({"case": name, "ok": ok, "detail": detail[:300]}), flush=True)
    ran = [c for c in doc["cases"].values() if c["ok"] is not None]
    doc["passed"] = all(c["ok"] for c in ran) and bool(ran)
    json.dump(doc, open(A.out, "w"), indent=1, ensure_ascii=False)
    sys.exit(0 if doc["passed"] else 1)


if __name__ == "__main__":
    main()
