#!/usr/bin/env python3
"""check_samples.py ENCODING_DIR KIND TEXT SAMPLES.jsonl: grammar_tool samples read by independent parsers.

KIND calls: each ended sample (the reply text, then EOS) goes through the official DeepSeek-V4.1 encoder's
parse_message_from_completion_text (encoding/encoding.py at the pinned checkpoint revision, chat mode); it must yield
calls only (empty content), each call's name one of the offered tools (namespace::name as encoded), its arguments a
JSON object holding the tool's required parameters with values of their declared types.
KIND json / json_schema: each ended sample must parse with Python's json module (and for json_schema validate against
TEXT with the small validator below: type, properties, required, additionalProperties false, items, enum, const,
min/max Length/Items, minimum/maximum, anyOf).
Prints a JSON summary; exit 0 only if every ended sample passes and at least one ended."""
import json
import sys

enc_dir, kind, text, path = sys.argv[1:5]
sys.path.insert(0, enc_dir)
import encoding  # noqa: E402

TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def valid(schema, x):
    if not isinstance(schema, dict):
        return True
    if "anyOf" in schema:
        return any(valid(s, x) for s in schema["anyOf"])
    if "enum" in schema:
        return x in schema["enum"]
    if "const" in schema:
        return x == schema["const"]
    t = schema.get("type")
    if t is not None:
        names = t if isinstance(t, list) else [t]
        ok = False
        for n in names:
            if n == "integer":
                ok |= isinstance(x, int) and not isinstance(x, bool)
            elif n == "number":
                ok |= isinstance(x, (int, float)) and not isinstance(x, bool)
            elif n == "boolean":
                ok |= isinstance(x, bool)
            else:
                ok |= isinstance(x, TYPES[n]) and not (n == "object" and isinstance(x, bool))
        if not ok:
            return False
    if isinstance(x, dict):
        props = schema.get("properties")
        for r in schema.get("required", []):
            if r not in x:
                return False
        for k, v in x.items():
            if props is not None and k not in props and schema.get("additionalProperties", False) is False:
                return False
            if props is not None and k in props and not valid(props[k], v):
                return False
    if isinstance(x, list):
        if len(x) < schema.get("minItems", 0) or len(x) > schema.get("maxItems", 1 << 60):
            return False
        if "items" in schema and not all(valid(schema["items"], v) for v in x):
            return False
    if isinstance(x, str):
        if len(x) < schema.get("minLength", 0) or len(x) > schema.get("maxLength", 1 << 60):
            return False
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if "minimum" in schema and x < schema["minimum"]:
            return False
        if "maximum" in schema and x > schema["maximum"]:
            return False
        if "exclusiveMinimum" in schema and x <= schema["exclusiveMinimum"]:
            return False
        if "exclusiveMaximum" in schema and x >= schema["exclusiveMaximum"]:
            return False
    return True


def tool_names(tools):
    out = {}
    for t in tools:
        f = t.get("function", t)
        ns = t.get("namespace") or f.get("namespace")
        if isinstance(ns, dict):
            ns = ns.get("name")
        name = f["name"] if ns is None or "::" in f["name"] else f"{ns}::{f['name']}"
        out[name] = f.get("parameters") or {}
    return out


samples = [json.loads(line) for line in open(path)]
ended = [s for s in samples if s["ended"]]
failures = []
calls_seen = 0
for s in ended:
    reply = s["text"]
    try:
        if kind == "calls":
            tools = tool_names(json.loads(text))
            msg = encoding.parse_message_from_completion_text(reply + encoding.eos_token, thinking_mode="chat")
            assert msg["content"] == "", f"content {msg['content']!r}"
            assert msg["tool_calls"], "no calls"
            for c in msg["tool_calls"]:
                f = c["function"]
                ns = c.get("namespace")
                name = f"{ns}::{f['name']}" if ns else f["name"]
                assert name in tools, f"unknown tool {name}"
                args = json.loads(f["arguments"])
                assert isinstance(args, dict)
                params = tools[name]
                if params.get("properties"):
                    assert valid(params, args), f"arguments {args} against {params}"
                calls_seen += 1
        else:
            value = json.loads(reply)
            if kind == "json":
                assert isinstance(value, dict)
            else:
                assert valid(json.loads(text), value), "schema"
    except Exception as e:  # noqa: BLE001
        failures.append({"text": reply[:400], "error": repr(e)[:300]})
summary = {"kind": kind, "samples": len(samples), "ended": len(ended), "passed": len(ended) - len(failures), "failed": len(failures), "calls": calls_seen, "failures": failures[:5]}
print(json.dumps(summary, ensure_ascii=False))
sys.exit(0 if ended and not failures else 1)
