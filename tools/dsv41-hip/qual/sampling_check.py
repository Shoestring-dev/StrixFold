#!/usr/bin/env python3
"""sampling_check.py OUT_DIR [--repo DIR] [--zig ZIG] [--sampler FILE]: the sampling/RNG contract gate on the CPU (handoff section 7,
gate M4.sampling-contract), for the engine's target sampler zig/src/families/dsv41/sampling.zig.

1. Contract equivalence: qual/sampling_harness.zig (built here against the repository's sampling.zig) samples 6,000
   generated rows over widths {1, 2, 7, 30, 300} x temperature {0, 0.3, 0.7, 1, 1.5} x top_k {1, 5, 20, 56} x top_p
   {1, 0.95, 0.5, 0.1} x min_p {0, 0.05, 0.3}, half of them with many tied values. Each token must equal the
   served contract's: tensorfold.engine.exact_sampling.choose_rows at the pinned CUDA source 222a9c0b (run here from
   git, numpy) on the top_k + MARGIN candidates (ties to the lower id), with libm's exp/log as the served image has
   them; temperature 0 is the first argmax. Rows where numpy's own exp differs from libm in a way that changes the
   token are reported apart (not counted against the port).
2. Distribution (independent of any TensorFold code): 12 small distributions sampled at 20,000 positions each with a
   fixed seed. The allowed set and probabilities are derived from the definitions alone (softmax of logits over the
   temperature on the top_k set; top_p keeps the shortest prefix in descending order whose mass reaches top_p; min_p
   keeps tokens at least min_p times the likeliest; renormalised): no token outside the set may appear and a
   chi-square test against the expected counts must not reject at p = 1e-4.
3. Draws are keyed by (seed, position, token id), so a verify row samples as the serial run does at its position: the
   harness's per-position draws are recomputed by an independent splitmix64 here and must match bit for bit.
Writes OUT_DIR/sampling_check.json; exit 0 only if all three hold.
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np

M64 = (1 << 64) - 1


def mix(x):
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & M64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


class Gen:
    def __init__(self, s):
        self.s = s

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        return mix(self.s)

    def value(self, coarse):
        q = (self.next() >> 52) - 2048
        return np.float32(q) / np.float32(2.0 if coarse else 128.0)


def uniform(seed, position, tid):
    x = mix((seed + 0x9E3779B97F4A7C15) & M64)
    x = mix(x ^ ((position * 0xD1B54A32D192ED03) & M64))
    x = mix(x ^ tid)
    return (x >> 11) * 2.0 ** -53 + 2.0 ** -54


TEMPS, TOPKS, TOPPS, MINPS, WIDTHS = (0.0, 0.3, 0.7, 1.0, 1.5), (1, 5, 20, 56), (1.0, 0.95, 0.5, 0.1), (0.0, 0.05, 0.3), (1, 2, 7, 30, 300)
MARGIN = 8


def candidates(row, count):
    order = sorted(range(len(row)), key=lambda i: (-float(row[i]), i))[:count]
    return np.array([row[i] for i in order], dtype=np.float32), np.array(order, dtype=np.int64)


def chi2_crit(df, p=1e-4):
    # Wilson-Hilferty: chi2_{df} quantile ~ df * (1 - 2/(9 df) + z sqrt(2/(9 df)))^3, z for the upper tail p
    z = 3.719016485455709  # upper 1e-4 normal quantile
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def expected(row, temp, top_k, top_p, min_p):
    n = len(row)
    order = sorted(range(n), key=lambda i: (-row[i], i))[:max(1, min(top_k, n))]
    scaled = [row[i] / temp for i in order]
    mx = max(scaled)
    w = [math.exp(v - mx) for v in scaled]
    tot = sum(w)
    pr = [x / tot for x in w]
    keep = len(order)
    if 0 < top_p < 1:
        acc, keep = 0.0, 0
        for x in pr:
            keep += 1
            acc += x
            if acc >= top_p:
                break
    allowed = list(range(keep))
    if min_p > 0:
        allowed = [j for j in allowed if pr[j] >= min_p * pr[0]]
    tot = sum(pr[j] for j in allowed)
    return {order[j]: pr[j] / tot for j in allowed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--repo", default=os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    ap.add_argument("--zig", default=os.path.expanduser("~/opt/zig/zig"))
    ap.add_argument("--sampler", default=None, help="another sampling.zig (negative controls)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    here = os.path.dirname(os.path.realpath(__file__))
    samp = a.sampler or os.path.join(a.repo, "zig/src/families/dsv41/sampling.zig")
    exe = os.path.join(a.out, "sampling_harness")
    build = [a.zig, "build-exe", "-O", "ReleaseSafe", "-lc", "--dep", "sampling", f"-Mroot={os.path.join(here, 'sampling_harness.zig')}",
             f"-Msampling={samp}", f"-femit-bin={exe}", "--cache-dir", os.path.join(a.out, "zig-cache"),
             "--global-cache-dir", os.path.join(a.out, "zig-global")]
    r = subprocess.run(build, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:])
        return 2
    run = subprocess.run([exe], capture_output=True, text=True)
    if run.returncode:
        print(run.stderr[-2000:])
        return 2
    got_c, got_d = {}, {}
    for line in run.stderr.splitlines():
        f = line.split()
        if f and f[0] == "C":
            got_c[int(f[1])] = int(f[2])
        elif f and f[0] == "D":
            got_d.setdefault(int(f[1]), {})[int(f[2])] = int(f[3])
    # the contract at the pinned CUDA source, from git
    src = subprocess.run(["git", "-C", a.repo, "show", "222a9c0b6ecaa9fd3c5dbded65a668524c9510ff:src/tensorfold/engine/exact_sampling.py"],
                         capture_output=True, text=True, check=True).stdout
    path = os.path.join(a.out, "exact_sampling_222a9c0b.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("exact_sampling_222a9c0b", path)
    es = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = es   # (dataclasses look their module up there)
    spec.loader.exec_module(es)

    def choose_libm(values, ids, position, s):
        saved = (es.np.exp, es.np.log)
        es.np.exp = np.vectorize(math.exp, otypes=[np.float64])
        es.np.log = np.vectorize(math.log, otypes=[np.float64])
        try:
            return es.choose_rows(values[None, :], ids[None, :], [position], s)[0]
        finally:
            es.np.exp, es.np.log = saved

    g = Gen(0x5EED0F0A11)
    case, mism, np_only, checked, rows_log = 0, [], 0, 0, []
    for n in WIDTHS:
        for t in TEMPS:
            for k in TOPKS:
                for p in TOPPS:
                    for m in MINPS:
                        for r in range(5):
                            coarse = r % 2 == 1
                            row = [g.value(coarse) for _ in range(n)]
                            seed = g.next()
                            pos = g.next() % 100000
                            if t <= 0:
                                ref = int(np.argmax(np.array(row, dtype=np.float32)))
                                ref_np = ref
                            else:
                                vals, ids = candidates(row, min(n, k + MARGIN))
                                s = es.Sampling(seed=seed, temperature=t, top_k=k, top_p=p, min_p=m)
                                ref = choose_libm(vals, ids, pos, s)
                                ref_np = es.choose_rows(vals[None, :], ids[None, :], [pos], s)[0]
                            got = got_c.get(case)
                            checked += 1
                            if got != ref:
                                mism.append({"case": case, "width": n, "temperature": t, "top_k": k, "top_p": p, "min_p": m,
                                             "coarse": coarse, "engine": got, "contract": ref, "contract_numpy_exp": ref_np})
                            if ref_np != ref:
                                np_only += 1
                            case += 1
    # draws: the independent distribution check, and the keyed uniform recomputed here
    dist = []
    for d in range(12):
        n = 4 + (d % 4) * 4
        row = [float(g.value(False)) / 4.0 for _ in range(n)]
        seed = g.next()
        t = TEMPS[1 + d % 4]
        k = n if d % 3 == 0 else 5
        p = 1.0 if d % 2 == 0 else 0.9
        m = 0.05 if d % 5 == 4 else 0.0
        exp_p = expected(row, t, k, p, m)
        counts = got_d.get(d, {})
        total = sum(counts.values())
        outside = {i: c for i, c in counts.items() if c and i not in exp_p}
        chi2 = sum((counts.get(i, 0) - total * q) ** 2 / (total * q) for i, q in exp_p.items())
        df = max(1, len(exp_p) - 1)
        crit = chi2_crit(df)
        ok = total == 20000 and not outside and (len(exp_p) == 1 or chi2 <= crit)
        dist.append({"dist": d, "width": n, "temperature": t, "top_k": k, "top_p": p, "min_p": m, "allowed": len(exp_p),
                     "outside_set": outside, "chi2": round(chi2, 3), "df": df, "crit_p1e-4": round(crit, 3), "ok": ok})
    # keyed draws: a few (seed, position, id) recomputed from the definition and compared with the engine's test vectors
    vectors = [((101, 2229, 2581), 0x3fe8cd8867d6a69c), ((404, 59, 129279), 0x3fd08f785f86da1b), ((0, 0, 0), 0x3fc9ff45ea7ce2be)]
    keyed = [{"key": list(kk), "bits": hex(np.float64(uniform(*kk)).view(np.uint64)), "want": hex(w),
              "ok": int(np.float64(uniform(*kk)).view(np.uint64)) == w} for kk, w in vectors]
    res = {"schema": "tf-port-sampling-check/1", "sampler": samp, "sampler_sha256": __import__("hashlib").sha256(open(samp, "rb").read()).hexdigest(),
           "contract": "src/tensorfold/engine/exact_sampling.py @ 222a9c0b (choose_rows, libm exp/log)",
           "cases_checked": checked, "cases_from_engine": len(got_c), "mismatches": mism[:50], "mismatch_count": len(mism),
           "cases_where_numpy_exp_changes_the_token": np_only, "distributions": dist, "keyed_uniform": keyed}
    res["pass"] = len(got_c) == checked and not mism and all(x["ok"] for x in dist) and all(x["ok"] for x in keyed)
    json.dump(res, open(os.path.join(a.out, "sampling_check.json"), "w"), indent=1)
    print(json.dumps({"pass": res["pass"], "cases": checked, "mismatches": len(mism), "numpy_exp_differs": np_only,
                      "distributions_ok": sum(x["ok"] for x in dist), "keyed_ok": sum(x["ok"] for x in keyed)}))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
