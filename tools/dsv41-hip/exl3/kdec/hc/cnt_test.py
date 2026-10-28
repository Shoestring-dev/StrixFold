"""cnt_test.py: _hc_fused's arrival counters after a long graph replay at mixed widths. One HIP graph of 120 fused
launches (rows 1..6 in turn over 20 sites' real weights), replayed 10 times = 1,200 consecutive launches; then every
counter must read 0, and each width's outputs must equal the 3-kernel chain's bit for bit."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "tfp/tri/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import triton
import tri_kernels as K
import hc_fuse as F

dev = "cuda"
D, NB, HC, WIDE = 5120, 40, 4, 20480
eps, hc_eps, iters = 1e-20, 1e-6, 20
man = {}
for line in open(Path.home() / "tfp/exl3/tensors/manifest_full.tsv"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 6:
        man[p[0]] = p


def tensor(name):
    p = man[name]
    with open(p[1], "rb") as f:
        f.seek(int(p[2]))
        b = f.read(int(p[3]))
    return torch.frombuffer(bytearray(b), dtype={"F32": torch.float32, "BF16": torch.bfloat16}[p[4]]).clone().to(dev)


def site(l, kind):
    pre = f"layers.{l}."
    return {"fn": tensor(pre + f"hc_{kind}_fn"), "base": tensor(pre + f"hc_{kind}_base"), "scale": tensor(pre + f"hc_{kind}_scale"),
            "nw": tensor(pre + ("attn_norm.weight" if kind == "attn" else "ffn_norm.weight"))}


def chain(s, x, pre_in, out, pre_out, post, comb, part, R):
    rb = F.hc_rb(R)
    K._hc_finish_s[(R, 1)](x, part, s["base"], s["scale"], pre_in, s["nw"], out, pre_out, post, comb, eps, hc_eps,
                           out, out, out, out, out, out, out, out, D=D, NB=NB, ITERS=iters, BLOCK=1024, NROT=0, RSPLIT=1,
                           num_warps=8, SINK=False, PDL=False)
    K._hc_mix_part[(triton.cdiv(R, rb), NB, 3)](x, x, x, 0, post, comb, s["fn"], part, R, WIDE=WIDE, D=D, NB=NB, SUB=128,
                                                MB=8, WORLD=1, POSTED=False, RB=rb, num_warps=2, PDL=False)
    K._hc_sinkhorn[(R,)](part, s["base"], s["scale"], pre_out, post, comb, eps, hc_eps, D=D, NB=NB, ITERS=iters, num_warps=8)


def fused(s, x, pre_in, out, pre_out, post, comb, part, cnt, R):
    rb = F.hc_rb(R)
    F._hc_fused[(triton.cdiv(R, rb), NB + 1, 3)](x, s["fn"], part, cnt, s["base"], s["scale"], pre_in, s["nw"], out, pre_out,
                                                post, comb, eps, hc_eps, R, WIDE=WIDE, D=D, NB=NB, SUB=128, MB=8, RB=rb,
                                                ITERS=iters, BLOCK=1024, SST=True, num_warps=8)


g = torch.Generator(device="cpu").manual_seed(5)
sites = [site(l, k) for l in range(20, 30) for k in ("attn", "ffn")]
data = {}
for R in range(1, 7):
    x = (torch.randn(R, HC, D, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    pre_in = (torch.rand(R, HC, generator=g) * 0.8 + 0.1).to(dev)
    outs = [torch.zeros((R, D), dtype=torch.bfloat16, device=dev), torch.zeros((R, 4), device=dev),
            torch.zeros((R, 4), device=dev), torch.zeros((R, 16), device=dev), torch.zeros(R * NB * 32, device=dev)]
    data[R] = (x, pre_in, outs)
cnt = torch.zeros(64, dtype=torch.int32, device=dev)
st = torch.cuda.Stream()
st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    for R in range(1, 7):
        fused(sites[0], data[R][0], data[R][1], *data[R][2], cnt, R)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
order = []
with torch.cuda.graph(graph, stream=st):
    for i in range(120):
        R = 1 + i % 6
        s = sites[i % len(sites)]
        order.append((R, i % len(sites)))
        fused(s, data[R][0], data[R][1], *data[R][2], cnt, R)
for _ in range(10):
    graph.replay()
torch.cuda.synchronize()
left = cnt.tolist()
res = {"launches": 1200, "counters_nonzero": sum(1 for v in left if v != 0), "counters_sum": sum(left), "widths": {}}
# each width's last launch in the graph (site index) against the chain on the same inputs
for R in range(1, 7):
    si = [j for (r, j) in order if r == R][-1]   # the site of the last launch of this width
    ref = [torch.full_like(t, 7) for t in data[R][2]]
    chain(sites[si], data[R][0], data[R][1], *ref, R)
    torch.cuda.synchronize()
    d = {}
    for nm, u, v in zip(("out", "pre_out", "post", "comb", "part"), data[R][2], ref):
        if nm == "part":
            u, v = u.view(R, NB, 32)[:, :, :25], v.view(R, NB, 32)[:, :, :25]
        view = torch.int16 if u.dtype == torch.bfloat16 else torch.int32
        d[nm] = int((u.contiguous().view(view) != v.contiguous().view(view)).sum())
    res["widths"][R] = d
res["pass"] = res["counters_nonzero"] == 0 and all(all(v == 0 for v in d.values()) for d in res["widths"].values())
print(json.dumps(res))
