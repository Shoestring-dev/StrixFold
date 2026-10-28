# l-link: grade split-on against split-off, teacher-forced, from two TF_DS_WINDOW_LOGITS dump directories (same
# prompts in the same order). Per position (every window row but the last, whose next token is not in the prompt):
# KL(off || on), top-1 agreement, NLL of the prompt's next token under each; paired NLL difference with a 95% CI.
# usage: llink_tfgrade.py OFF_DIR ON_DIR
import sys, os, glob, json
import numpy as np
def load(p):
    b = open(p, "rb").read()
    n, vocab = np.frombuffer(b, np.uint32, 2, 0)
    start = int(np.frombuffer(b, np.uint64, 1, 8)[0])
    ids = np.frombuffer(b, np.int32, n + 1, 16)
    lg = np.frombuffer(b, np.float32, n * vocab, 16 + (n + 1) * 4).reshape(n, vocab)
    return int(n), int(vocab), start, ids, lg
def lsm(x):
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))
off_files = sorted(glob.glob(os.path.join(sys.argv[1], "*.bin")))
on_files = sorted(glob.glob(os.path.join(sys.argv[2], "*.bin")))
assert len(off_files) == len(on_files) and off_files, (len(off_files), len(on_files))
KL, AG, D, per = [], [], [], []
for fo, fn in zip(off_files, on_files):
    n, v, s, ids, a = load(fo)
    n2, v2, s2, ids2, b = load(fn)
    assert (n, v, s) == (n2, v2, s2) and (ids == ids2).all(), (fo, fn)
    la, lb = lsm(a[:-1]), lsm(b[:-1])
    tgt = ids[1:n]
    kl = (np.exp(la) * (la - lb)).sum(axis=1)
    ag = la.argmax(axis=1) == lb.argmax(axis=1)
    d = -lb[np.arange(n - 1), tgt] + la[np.arange(n - 1), tgt]
    same = bool((a == b).all())
    KL.append(kl); AG.append(ag); D.append(d)
    per.append({"file": os.path.basename(fo), "start": s, "positions": n - 1, "bit_identical": same, "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                "top1": float(ag.mean()), "dnll_mean": float(d.mean()), "nll_off": float((-la[np.arange(n - 1), tgt]).mean())})
KL, AG, D = np.concatenate(KL), np.concatenate(AG), np.concatenate(D)
N = len(D)
ci = 1.96 * D.std(ddof=1) / np.sqrt(N)
res = {"prompts": len(per), "positions": N, "kl_mean": float(KL.mean()), "kl_p99": float(np.quantile(KL, 0.99)), "kl_max": float(KL.max()),
       "top1_agreement": float(AG.mean()), "dnll_mean": float(D.mean()), "dnll_ci95_upper": float(D.mean() + ci),
       "pass": bool(N >= 2048 and KL.mean() <= 6e-3 and AG.mean() >= 0.96 and D.mean() + ci <= 0.005)}
for p in per: print(json.dumps(p))
print(json.dumps(res))
