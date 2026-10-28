#!/usr/bin/env python3
"""par_prompts.py KIND OUT [--model DIR] [--texts DIR]: the token-id request files of the served-config parity checks
(PLAN-research-actions.md section 1), from real tokenized text with the bundle's tokenizer.json (and its
chat_template.jinja for the chat prompts). Needs the `tokenizers` and `jinja2` packages (node1: ~/venvs/tfp-inputs).

Every prompt is [BOS 0] + the text's ids (add_special_tokens off), cut to an exact length, as w-ref's replay_gap_long.py
makes its prompts; the chat prompts are the template's rendering (thinking off, as the server's --no-thinking) of the
served bench's code and prose prompts and of long_prompt.py's 2K/4K/8K documents, tokenized whole. Every length is
asserted.

KIND
  a1    tf-dsv41-lanes --requests for A1: the determinism control [A, X, A', X] (X: 2,148 tokens, a 100-row last
        chunk), then [A, B, A', B] for each P_B in 2050, 2062, 2064, 4100 (A, A': two different 3,000-token prompts;
        B: the same text cut to P_B), then the reference rows ref2050 and ref4096 (w-ref's prose-4096-fp64 ids, the
        2050 ones its first 2,050). max_tokens 4, greedy.
  a23   tf-dsv41-lanes --requests for the A2/A3 lane checks: the determinism control, the 2K/4K/8K chat prompts
        (max_tokens 4: their prompt shas, paio) and the code and prose chat prompts (max_tokens 400: their reply
        token_shas, aio and side), greedy.
TEXTS (--texts, default ~/tfp/par/texts): code_ds41_inference.txt, readme_model.txt, prose_pg1342_body.txt,
long_memory_archive.txt (long_prompt.py's source) and ref4096-ids.json (node0 ~/tfp/ref/out/gap/prose-4096-fp64
state.json's first 4,096 ids)."""
import argparse
import hashlib
import json
import os

P_B = (2050, 2062, 2064, 4100)
A_LEN, X_LEN = 3000, 2148
BENCH = {  # bench_stream.py's decode prompts
    "code": "Write a Python class implementing an LRU cache with get/put in O(1), with type hints and a short usage example.",
    "prose": "Explain to a new manager, in about 300 words, why code review slows teams down in the short term but speeds them up over a year.",
}
LONG = ((11000, "2k", 2010), (22000, "4k", 4002), (45000, "8k", 8165))  # long_prompt.py's cuts; the served counts
LONG_Q = "Summarise the main events described above in a few sentences."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("a1", "a23"))
    ap.add_argument("out")
    ap.add_argument("--model", default=os.path.expanduser("~/models/tf-port/inputs-tp2-c178a012/model"))
    ap.add_argument("--texts", default=os.path.expanduser("~/tfp/par/texts"))
    a = ap.parse_args()
    import tokenizers
    tok = tokenizers.Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))

    def text(name, at=0):
        return open(os.path.join(a.texts, name), encoding="utf-8").read()[at:]

    def cut(name, n, at=0):
        ids = [0] + tok.encode(text(name, at)[: 8 * n], add_special_tokens=False).ids
        assert len(ids) >= n, (name, at, n, len(ids))
        return ids[:n]

    def req(name, ids, max_tokens, rows=None):
        assert rows is None or len(ids) == rows, (name, len(ids), rows)
        return {"name": name, "prompt": ids, "max_tokens": max_tokens}

    A = cut("code_ds41_inference.txt", A_LEN)
    A2 = cut("readme_model.txt", A_LEN)
    X = cut("prose_pg1342_body.txt", X_LEN, at=500000)
    assert A != A2
    det = [req("det.A", A, 4, A_LEN), req("det.X1", X, 4, X_LEN), req("det.A'", A2, 4, A_LEN), req("det.X2", X, 4, X_LEN)]
    reqs = list(det)
    meta = {"A": A_LEN, "X": X_LEN}
    if a.kind == "a1":
        B = cut("prose_pg1342_body.txt", max(P_B), at=300000)
        for p in P_B:
            b = B[:p]
            reqs += [req(f"B{p}.A", A, 4, A_LEN), req(f"B{p}.afterA", b, 4, p), req(f"B{p}.A'", A2, 4, A_LEN), req(f"B{p}.afterA'", b, 4, p)]
        ref = json.load(open(os.path.join(a.texts, "ref4096-ids.json")))["ids"]
        assert len(ref) == 4096 and ref == cut("prose_pg1342_body.txt", 4096), "ref4096 ids differ from this tokenization"
        reqs += [req("ref2050", ref[:2050], 4, 2050), req("ref4096", ref, 4, 4096)]
        meta.update({"P_B": list(P_B), "ref": "node0 ~/tfp/ref/out/gap/prose-4096-fp64/state.json ids[:4096] (2050: ids[:2050])"})
    else:
        import jinja2
        tpl = jinja2.Environment().from_string(open(os.path.join(a.model, "chat_template.jinja"), encoding="utf-8").read())

        def chat(content):
            s = tpl.render(messages=[{"role": "user", "content": content}], add_generation_prompt=True, enable_thinking=False)
            return tok.encode(s, add_special_tokens=False).ids

        src = text("long_memory_archive.txt")
        for chars, name, served in LONG:
            doc = ((src + "\n") * 3)[:chars]
            ids = chat(doc + "\n\nQuestion: " + LONG_Q)
            assert len(ids) == served, (name, len(ids), served)
            reqs.append(req(f"long{name}", ids, 4, served))
        for name, served in (("code", 31), ("prose", 34)):
            ids = chat(BENCH[name])
            assert len(ids) == served, (name, len(ids), served)
            reqs.append(req(name, ids, 400, served))
        meta.update({"long": [x[2] for x in LONG], "decode": {"code": 31, "prose": 34}})
    sha = {f: hashlib.sha256(open(os.path.join(a.texts, f), "rb").read()).hexdigest() for f in sorted(os.listdir(a.texts))}
    doc = {"kind": a.kind, "texts": sha, "tokenizer_sha256": hashlib.sha256(open(os.path.join(a.model, "tokenizer.json"), "rb").read()).hexdigest(),
           "meta": meta, "requests": reqs}
    json.dump(doc, open(a.out, "w"))
    print(json.dumps({"out": a.out, "requests": [(r["name"], len(r["prompt"]), r["max_tokens"]) for r in reqs]}))


if __name__ == "__main__":
    main()
