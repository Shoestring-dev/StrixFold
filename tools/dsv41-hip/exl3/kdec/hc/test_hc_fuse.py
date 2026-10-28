"""k-dec2 hc fusion, stage 1: the fused kernel (hc_fuse._hc_fused) against the served chain (_hc_finish_s role 1,
_hc_mix_part 2 warps, _hc_sinkhorn), bit for bit, then timed back to back as HIP graphs at served clocks.
Weights: the checkpoint's own (hc_{attn,ffn}_fn / base / scale, {attn,ffn}_norm) of layers 20 (attn), 39 (ffn), 1
(attn). Activations: the streams and the carried pre-mix are synthetic (bf16 N(0, 0.5) streams, pre in 0.1-0.9; the
served round's buffers are not dumped by any engine tool today). Rows 1, 3, 5, 6."""
import json, struct, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.home() / "tfp/tri/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import triton
import tri_kernels as K
import hc_fuse as F

dev = "cuda"
D, NB, HC = 5120, 40, 4
WIDE = HC * D
eps, hc_eps, iters = 1e-20, 1e-6, 20
man = {}
for line in open(Path.home() / "tfp/exl3/tensors/manifest_full.tsv"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 6:
        man[p[0]] = p


def tensor(name, dtype):
    p = man[name]
    with open(p[1], "rb") as f:
        f.seek(int(p[2]))
        b = f.read(int(p[3]))
    t = torch.frombuffer(bytearray(b), dtype={"F32": torch.float32, "BF16": torch.bfloat16}[p[4]]).clone()
    return t.to(dev)


def site(l, kind):
    pre = f"layers.{l}."
    return {"fn": tensor(pre + f"hc_{kind}_fn", None).contiguous(), "base": tensor(pre + f"hc_{kind}_base", None),
            "scale": tensor(pre + f"hc_{kind}_scale", None),
            "nw": tensor(pre + ("attn_norm.weight" if kind == "attn" else "ffn_norm.weight"), None)}


def chain(s, x, pre_in, out, pre_out, post, comb, part, R):
    rb = F.hc_rb(R)
    K._hc_finish_s[(R, 1)](x, part, s["base"], s["scale"], pre_in, s["nw"], out, pre_out, post, comb, eps, hc_eps,
                           out, out, out, out, out, out, out, out, D=D, NB=NB, ITERS=iters, BLOCK=1024, NROT=0, RSPLIT=1,
                           num_warps=8, SINK=False, PDL=False)
    K._hc_mix_part[(triton.cdiv(R, rb), NB, 3)](x, x, x, 0, post, comb, s["fn"], part, R, WIDE=WIDE, D=D, NB=NB,
                                                SUB=128, MB=8, WORLD=1, POSTED=False, RB=rb, num_warps=2, PDL=False)
    K._hc_sinkhorn[(R,)](part, s["base"], s["scale"], pre_out, post, comb, eps, hc_eps, D=D, NB=NB, ITERS=iters,
                         num_warps=8)


SST = int(__import__("os").environ.get("SST", "1"))
def fused(s, x, pre_in, out, pre_out, post, comb, part, cnt, R):
    rb = F.hc_rb(R)
    F._hc_fused[(triton.cdiv(R, rb), NB + 1, 3)](x, s["fn"], part, cnt, s["base"], s["scale"], pre_in, s["nw"], out,
                                                pre_out, post, comb, eps, hc_eps, R, WIDE=WIDE, D=D, NB=NB, SUB=128,
                                                MB=8, RB=rb, ITERS=iters, BLOCK=1024, SST=bool(SST), num_warps=8)


def bufs(R, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(R, HC, D, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    pre_in = (torch.rand(R, HC, generator=g) * 0.8 + 0.1).to(dev)
    return x, pre_in


def outs(R):
    return (torch.full((R, D), 7, dtype=torch.bfloat16, device=dev), torch.full((R, 4), 7.0, device=dev),
            torch.full((R, 4), 7.0, device=dev), torch.full((R, 16), 7.0, device=dev),
            torch.zeros(R * NB * 32, device=dev))


res = {"bits": [], "timing": []}
sites = {"L20.attn": site(20, "attn"), "L39.ffn": site(39, "ffn"), "L1.attn": site(1, "attn")}
cnt = torch.zeros(64, dtype=torch.int32, device=dev)
ok_all = True
for name, s in sites.items():
    for R in (1, 3, 5, 6):
        for seed in range(3):
            x, pre_in = bufs(R, 1000 * R + seed)
            a = outs(R)
            b = outs(R)
            # the chain's posts / combs are inputs of nothing here (no post); the fused kernel must leave the same
            chain(s, x, pre_in, *a, R)
            fused(s, x, pre_in, *b, cnt, R)
            torch.cuda.synchronize()
            names = ("out", "pre_out", "post", "comb", "part")
            diffs = {}
            for nm, u, v in zip(names, a, b):
                if nm == "part":
                    u = u.view(R, NB, 32)[:, :, :25]
                    v = v.view(R, NB, 32)[:, :, :25]
                diffs[nm] = int((u.contiguous().view(torch.int16 if u.dtype == torch.bfloat16 else torch.int32)
                                 != v.contiguous().view(torch.int16 if v.dtype == torch.bfloat16 else torch.int32)).sum())
            pa, pb = a[4].view(R, NB, 32)[:, :, :25], b[4].view(R, NB, 32)[:, :, :25]
            diffs["part_slots"] = sorted({int(i) for i in torch.nonzero(pa.contiguous().view(torch.int32) != pb.contiguous().view(torch.int32))[:, 2].tolist()})
            ok = all(d == 0 for kk, d in diffs.items() if kk != "part_slots") and int(cnt.sum()) == 0
            ok_all &= ok
            res["bits"].append({"site": name, "rows": R, "seed": seed, "diffs": diffs, "cnt_left": int(cnt.sum()), "ok": ok})
            print(json.dumps(res["bits"][-1]), flush=True)

# timing: 80 sites' worth of launches captured as a graph (20 distinct weight sets x 4, so the weights are cold as
# they are in a round), replayed back to back; per site
rng = [site(l, k) for l in range(2, 12) for k in ("attn", "ffn")]
for R in (1, 3, 6):
    x, pre_in = bufs(R, 7)
    a = outs(R)
    for kind in ("chain", "fused", "chain", "fused"):
        g = torch.cuda.CUDAGraph()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for i in range(4):  # warm / compile outside the capture
                (chain if kind == "chain" else fused)(rng[i], x, pre_in, *a, *(() if kind == "chain" else (cnt,)), R)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=st):
            for i in range(80):
                s = rng[i % len(rng)]
                if kind == "chain":
                    chain(s, x, pre_in, *a, R)
                else:
                    fused(s, x, pre_in, *a, cnt, R)
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(15):
            e0.record(); g.replay(); e1.record(); e1.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3 / 80)
        ts.sort()
        res["timing"].append({"rows": R, "kind": kind, "us_per_site": round(ts[len(ts) // 2], 2), "min": round(ts[0], 2)})
        print(json.dumps(res["timing"][-1]), flush=True)
        del g
print(json.dumps({"bits_ok": ok_all}))
