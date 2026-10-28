#!/usr/bin/env python3
"""budget.py: a served decode round's time budget from rocprofv3 kernel traces (WRAP0 / WRAP1), one or both ranks.

budget.py [--r0 CSV] [--r1 CSV] [--sclk0 JSONL] [--sclk1 JSONL] [--manifest TSV] [--rows 1] [--json OUT]

Rounds: each decode round starts at its _embed_init launch (rows R = grid x / 128); a round's wall time is from its
first launch (the uploads before _embed_init) to the next round's first launch. Inside a round the gather sites are
the tf_ds_coll_collect launches in order: per layer an Engram gather (layers 1, 14), the attention output's and the
MoE's; then the head's; anything after the head's gather is the drafter (MTP block, its head) and the round's tail.

Per round and rank:
- weight streaming: glinear (dense, by group), experts (grouped_cp epi1 + epi2, decode_prep apart), the router's
  _rowmm_parts; bytes from the checkpoint manifest (TP2 rank slices), time at 230 GB/s and the excess over it;
- small kernels: everything else that is not a gather;
- gathers: stage + collect per site; a collect's wait = its time minus the site's floor (the 5th percentile of that
  site's collects over the run, i.e. the copy with the peer's data already there);
- gaps: between consecutive launches of the round below 20 us (graph / launch gaps), and from 20 us (the host:
  stretch boundaries where the GPU waits on the host); the GPU idle from the round's last kernel to the next round.
"""
import argparse, bisect, collections, json, math, re, statistics, sys

BW = 230e9  # bytes/s, streaming read ceiling (k-dec bw.hip)
GAP_HOST_US = 20.0


def load(path):
    import csv
    ks = []
    with open(path) as f:
        r = csv.DictReader(f)
        for x in r:
            if x.get("Kind", "KERNEL_DISPATCH") != "KERNEL_DISPATCH":
                continue
            ks.append((int(x["Start_Timestamp"]), int(x["End_Timestamp"]), x["Kernel_Name"], int(x["Grid_Size_X"]),
                       int(x["Grid_Size_Y"]), int(x["Grid_Size_Z"]), int(x["Queue_Id"])))
    ks.sort()
    return ks


# ---- the checkpoint's tensors, per TP2 rank (weights.zig / plan.zig Split, world 2, no pair)
def manifest(path):
    t = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) < 6 or not p[0].endswith(".trellis"):
            continue
        a, b, c = (int(v) for v in p[5].split("x"))
        t[p[0][:-8]] = (a * 16, b * 16, c // 8)  # k, n, k2 (bits = k2 / 2)
    return t


def plan(k, n):  # exl3_linear.zig plan()
    blocks, min_tiles, kt, nb = 192, 8, k // 16, n // 128
    sk, wk = 1, (8 if nb >= 64 else 4)
    if nb < 64:
        while nb * sk < blocks and sk < 64:
            sk *= 2
    while (wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt // (sk * wk) < min_tiles):
        if wk > 4:
            wk = 4
        elif sk > 1:
            sk //= 2
        else:
            break
    return sk, wk


def plan_hip(k, n):  # kdec 77df5c7 planHip
    sk, wk = plan(k, n)
    for tk, tn, ts in ((5120, 1280, 4), (5120, 512, 4), (1280, 4096, 1)):
        if k == tk and n == tn and ts < sk:
            sk = ts
    return sk, wk


def rank_groups(T, prefix):
    """A layer's dense EXL3 linears as one TP2 rank holds them, in the round's launch groups and order (round.zig):
    engram wkv; wq_a / wkv / compressor wkv, wgate; wq_b / indexer wq_b; indexer wk; wo_a's 4 groups; wo_b; (drafter:
    main_proj first). Each entry: (name, k, n, k2)."""
    def get(name, k=None, n=None):
        if name not in T:
            return []
        kk, nn, k2 = T[name]
        return [(name, k or kk, n or nn, k2)]
    g = []
    g.append(get(prefix + ".main_proj"))
    if prefix + ".engram.wkv" in T:
        g.append(get(prefix + ".engram.wkv", k=T[prefix + ".engram.wkv"][0] // 2))
    g.append(get(prefix + ".attn.wq_a") + get(prefix + ".attn.wkv") + get(prefix + ".attn.compressor.wkv") +
             get(prefix + ".attn.compressor.wgate"))
    g.append((get(prefix + ".attn.wq_b", n=T[prefix + ".attn.wq_b"][1] // 2) if prefix + ".attn.wq_b" in T else []) +
             get(prefix + ".attn.indexer.wq_b"))
    g.append(get(prefix + ".attn.indexer.wk"))
    g.append(sum((get(prefix + ".attn.wo_a.slice.%d" % i) for i in range(4)), []))
    g.append(get(prefix + ".attn.wo_b", k=T[prefix + ".attn.wo_b"][0] // 2) if prefix + ".attn.wo_b" in T else [])
    return [x for x in g if x]


class GroupWalker:
    """Matches a section's glinear launches in order against its launch groups."""
    def __init__(self, groups, planf):
        self.groups = [list(x) for x in groups]
        self.gi = 0
        self.planf = planf

    def take(self, k2, wk, blocks):
        for gi in range(self.gi, len(self.groups)):
            got = match_glinear(self.groups[gi], k2, wk, blocks, self.planf)
            if got:
                for c in got:
                    self.groups[gi].remove(c)
                self.gi = gi
                return got
        return None


def lin_bytes(k, n, k2):
    return k * n * k2 // 16


def match_glinear(cands, k2, wk, blocks, planf):
    """The subset of cands (same k2 and warps) whose plan blocks sum to `blocks`: prefer the largest consistent set in
    order (a launch takes a contiguous run of calls of one (k2, warps) set)."""
    pool = [c for c in cands if c[3] == k2 and planf(c[1], c[2])[1] == wk]
    best = None
    n = len(pool)
    for mask in range(1, 1 << n):
        sel = [pool[i] for i in range(n) if mask >> i & 1]
        b = sum((c[2] // 128) * planf(c[1], c[2])[0] for c in sel)
        if b == blocks and (best is None or len(sel) > len(best)):
            best = sel
    return best


def expert_bytes(T, layer, mtp=False):
    p = ("mtp.%d" % layer) if mtp else ("layers.%d" % layer)
    k2r = T[p + ".ffn.experts.0.w1"][2]
    k2s = T[p + ".ffn.shared_experts.w1"][2]
    inter = T[p + ".ffn.experts.0.w1"][1] // 2  # the rank's half
    gu = lambda k2: 2 * lin_bytes(5120, inter, k2)
    dn = lambda k2: lin_bytes(inter, 5120, k2)
    return {"k2_routed": k2r, "k2_shared": k2s, "gu_routed": gu(k2r), "dn_routed": dn(k2r), "gu_shared": gu(k2s),
            "dn_shared": dn(k2s)}


def kind(name):
    if name.startswith("tf_ds_exl3_glinear"):
        return "glinear"
    if name.startswith("tf_ds_exl3_experts_grouped"):
        return "experts"
    if name.startswith("tf_ds_exl3_experts_decode_prep"):
        return "experts_prep"
    if name.startswith("tf_ds_coll_collect"):
        return "collect"
    if name.startswith("tf_ds_coll_stage"):
        return "stage"
    if name.startswith("_rowmm_parts"):
        return "router"
    if name.startswith("__amd_rocclr"):
        return "copy"
    return "small"


def segment(ks):
    """Segments: [start, embed index, end, rows, collects] from each _embed_init (the uploads before it included). A
    main round has >= 80 gathers (83: 40 layers x 2, Engram 2, head); a drafter pass (DSpark block) fewer."""
    emb = [i for i, k in enumerate(ks) if k[2].startswith("_embed_init")]
    segs = []
    for ei in emb:
        st = ei
        while st > 0 and ks[st - 1][2].startswith("__amd_rocclr") and ks[st][0] - ks[st - 1][1] < 1_000_000:
            st -= 1
        if segs:
            segs[-1][2] = st
        segs.append([st, ei, len(ks), ks[ei][3] // 128, 0])
    for s in segs:
        s[4] = sum(1 for k in ks[s[0]:s[2]] if k[2].startswith("tf_ds_coll_collect"))
    return segs


def round_detail(ks, st, en, T, planf, layers_main=40):
    """Per-launch attribution of one round: list of dicts."""
    seg = ks[st:en]
    recs = []
    site = 0
    layer = 0
    phase = "pre"
    # gather sites in order: main model: for l in 0..39: [engram if l in (1,14)], attn, moe; then head; then drafter
    order = []
    for l in range(layers_main):
        if l in (1, 14):
            order.append(("L%d" % l, "engram"))
        order.append(("L%d" % l, "attn"))
        order.append(("L%d" % l, "moe"))
    order.append(("head", "head"))
    prev_end = None
    for i, k in enumerate(seg):
        s, e, name, gx, gy, gz, q = k
        gap = (s - prev_end) / 1e3 if prev_end is not None else 0.0
        prev_end = max(prev_end or 0, e)
        where = order[site] if site < len(order) else ("draft", "d%d" % (site - len(order)))
        recs.append({"i": i, "name": name, "kind": kind(name), "dur": (e - s) / 1e3, "gap": gap, "gx": gx, "gy": gy,
                     "site": site, "layer": where[0], "part": where[1], "q": q, "s": s, "e": e})
        if kind(name) == "collect":
            site += 1
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0"); ap.add_argument("--r1")
    ap.add_argument("--sclk0"); ap.add_argument("--sclk1")
    ap.add_argument("--manifest", default="manifest_full.tsv")
    ap.add_argument("--rows", default="1")
    ap.add_argument("--plan", default="plan", choices=["plan", "hip"])
    ap.add_argument("--json")
    ap.add_argument("--skip", type=int, default=3, help="decode rounds skipped at each request's start")
    a = ap.parse_args()
    T = manifest(a.manifest)
    planf = plan if a.plan == "plan" else plan_hip
    rows_want = {int(x) for x in a.rows.split(",")} if a.rows != "all" else None
    report = {}
    for rk, path, sp in (("rank0", a.r0, a.sclk0), ("rank1", a.r1, a.sclk1)):
        if not path:
            continue
        ks = load(path)
        report[rk] = analyse(ks, T, planf, rows_want, a.skip, sp)
    if a.r0 and a.r1:
        report["pair"] = pair_sites(report)
    txt = render(report)
    print(txt)
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)


def analyse(ks, T, planf, rows_want, skip, sclk_path):
    segs = segment(ks)
    mains = [s for s in segs if s[4] >= 80]
    # requests: a gap of > 50 ms between rounds or a prompt chunk (rows > 8) starts a new request; the first `skip`
    # decode rounds of each are left out (graph captures)
    sel = []
    since = 0
    for j, (st, ei, en, R, nc) in enumerate(mains):
        if j + 1 >= len(mains):
            break
        nxt = mains[j + 1][0]
        if j == 0 or ks[st][0] - ks[mains[j - 1][2] - 1][1] > 50_000_000:
            since = 0
        if R > 8:
            since = 0
            continue
        since += 1
        if ks[nxt][0] - ks[en - 1][1] > 50_000_000:
            continue  # the request's last round: the next main round is another request's
        if rows_want is not None and R not in rows_want:
            continue
        if since <= skip:
            continue
        sel.append((R, st, nxt))
    # per-layer linears for byte attribution
    lin = {}
    for l in range(40):
        lin["L%d" % l] = rank_groups(T, "layers.%d" % l)
    head = [("head", 5120, T["head"][1] // 2, T["head"][2])]
    # the drafter (DSpark, draft.zig): main_proj, every stage's wkv as one group, then each stage's block
    st = [p for p in ("mtp.0", "mtp.1", "mtp.2") if p + ".attn.wkv" in T]
    lin["mtp"] = [[("mtp.0.main_proj",) + T["mtp.0.main_proj"]]] + [[(p + ".attn.wkv",) + T[p + ".attn.wkv"] for p in st]]
    for p in st:
        lin["mtp"] += [g for g in rank_groups(T, p) if not g[0][0].endswith("main_proj")]
    xb = {("L%d" % l): expert_bytes(T, l) for l in range(40)}
    xb["mtp"] = expert_bytes(T, 0, mtp=True)
    agg = collections.defaultdict(float)
    per_round = []
    site_coll = collections.defaultdict(list)
    site_stage = collections.defaultdict(list)
    site_gap = collections.defaultdict(list)
    unmatched = collections.Counter()
    glin_by = collections.defaultdict(lambda: [0.0, 0, 0])  # group label -> time, bytes, launches
    xp_by = collections.defaultdict(lambda: [0.0, 0, 0])
    sclk = load_sclk(sclk_path) if sclk_path else None
    sclk_by = collections.defaultdict(list)
    nround = 0
    for (R, st, en) in sel:
        recs = round_detail(ks, st, en, T, planf)
        wall = (ks[en][0] - ks[st][0]) / 1e3
        last_end = max(r["e"] for r in recs)
        tail_idle = (ks[en][0] - last_end) / 1e3
        walkers = {}
        b = collections.defaultdict(float)
        b["wall"] = wall
        b["idle_after"] = tail_idle
        iv = sorted((r["s"], r["e"]) for r in recs)
        busy, cs, ce = 0, iv[0][0], iv[0][1]
        for x0, x1 in iv[1:]:
            if x0 > ce:
                busy += ce - cs; cs, ce = x0, x1
            else:
                ce = max(ce, x1)
        busy += ce - cs
        b["busy"] = busy / 1e3
        nround += 1
        # draft section: after the head's collect
        for r in recs:
            sec = "draft" if r["layer"] == "draft" else "main"
            kd = r["kind"]
            if r["gap"] < GAP_HOST_US:
                b["gap_launch"] += r["gap"]
                if sec == "draft":
                    b["draft_gap_launch"] += r["gap"]
            else:
                b["gap_host"] += r["gap"]
                if sec == "draft":
                    b["draft_gap_host"] += r["gap"]
            if sec == "draft":
                b["draft_" + kd] += r["dur"]
                b["draft_total"] += r["dur"]
            else:
                b[kd] += r["dur"]
            if kd == "collect":
                site_coll[(r["layer"], r["part"])].append(r["dur"])
                site_gap[(r["layer"], r["part"])].append(r["gap"])
            if kd == "stage":
                site_stage[(r["layer"], r["part"])].append(r["dur"])
            if kd == "glinear":
                m = re.match(r"tf_ds_exl3_glinear1?_k(\d+)_w(\d+)", r["name"])
                k2, wk = int(m.group(1)), int(m.group(2))
                blocks = r["gx"] // (wk * 32)
                key = r["layer"] if r["layer"] != "draft" else "mtp"
                hb = (T["head"][1] // 2 // 128) * planf(5120, T["head"][1] // 2)[0]
                if r["layer"] == "head" or (key == "mtp" and blocks == hb and k2 == T["head"][2]):
                    got = match_glinear(head, k2, wk, blocks, planf)
                    key = "head" if key != "mtp" else "mtp"
                else:
                    wkey = key if key == "mtp" else key
                    if wkey not in walkers or (key == "mtp" and walkers[wkey].gi >= len(walkers[wkey].groups)):
                        walkers[wkey] = GroupWalker(lin.get(key, []), planf)
                    got = walkers[wkey].take(k2, wk, blocks)
                    if got is None and key == "mtp":  # the drafter's next step: its groups again
                        walkers[wkey] = GroupWalker(lin.get(key, []), planf)
                        got = walkers[wkey].take(k2, wk, blocks)
                if got is None:
                    unmatched[(key, k2, wk, blocks)] += 1
                    by = 0
                    label = "?%s_k%d_w%d_b%d" % (key, k2, wk, blocks)
                else:
                    by = sum(lin_bytes(c[1], c[2], c[3]) for c in got)
                    names = sorted({re.sub(r"\.slice\.\d+$", "", re.sub(r"^(layers|mtp)\.\d+\.", "", c[0])) + ("x%d" % sum(1 for x in got if x[0].endswith(c[0].split(".", 2)[-1])) if c[0].startswith("mtp") and sum(1 for x in got if x[0].endswith(c[0].split(".", 2)[-1])) > 1 else "") for c in got})
                    label = ("mtp:" if key == "mtp" else "") + "+".join(names)
                b["glinear_bytes" if sec == "main" else "draft_glinear_bytes"] += by
                g = glin_by[label]; g[0] += r["dur"]; g[1] += by; g[2] += 1
            if kd == "experts":
                key = r["layer"] if r["layer"] != "draft" else "mtp"
                x = xb.get(key)
                if x and R == 1:
                    epi = "gu" if "epi1" in r["name"] else "dn"
                    by = 6 * x[epi + "_routed"] + x[epi + "_shared"]
                    b["experts_bytes" if sec == "main" else "draft_experts_bytes"] += by
                    lab = ("mtp" if key == "mtp" else "k2r%d" % x["k2_routed"]) + "_" + epi
                    g = xp_by[lab]; g[0] += r["dur"]; g[1] += by; g[2] += 1
            if kd == "router" and sec == "main":
                b["router_bytes"] += 384 * 5120 * 2
            if sclk is not None and kd in ("experts", "glinear", "collect", "small"):
                v = sclk_at(sclk, r["s"], r["e"])
                if v is not None:
                    sclk_by[kd + ("_draft" if sec == "draft" else "")].append(v)
        per_round.append(dict(b, R=R))
        for k, v in b.items():
            agg[k] += v
    # prompt chunks (rows > 8): each chunk's kernels by kind, from its first launch to the next segment's
    prompts = []
    for j, (st, ei, en, R, nc) in enumerate(mains):
        if R <= 8:
            continue
        nxt = segs[[x[0] for x in segs].index(st) + 1][0] if [x[0] for x in segs].index(st) + 1 < len(segs) else len(ks)
        kk = ks[st:en]
        byk = collections.defaultdict(float)
        for k in kk:
            byk[kind(k[2])] += (k[1] - k[0]) / 1e3
        prompts.append({"rows": R, "span_us": (ks[en - 1][1] - ks[st][0]) / 1e3, "by_kind_us": dict(byk)})
    n = max(nround, 1)
    mean = {k: v / n for k, v in agg.items()}
    # per site: the floor (p5) and the mean wait above it
    sites = {}
    for key, v in site_coll.items():
        v = sorted(v)
        floor = v[max(0, int(0.05 * len(v)) - 1)] if v else 0
        st = site_stage.get(key, [])
        g = sorted(site_gap.get(key, [0.0]))
        sites["%s.%s" % key] = {"n": len(v), "mean": sum(v) / len(v), "median": v[len(v) // 2], "p5": floor,
                                "p90": v[min(len(v) - 1, int(0.9 * len(v)))], "stage_to_collect_gap_median": g[len(g) // 2],
                                "p95": v[min(len(v) - 1, int(0.95 * len(v)))], "wait_mean": sum(x - floor for x in v) / len(v),
                                "stage_mean": (sum(st) / len(st)) if st else 0.0}
    res = {"rounds": nround, "mean_us": mean, "sites": sites,
           "glinear_groups": {k: {"us_per_round": v[0] / n, "bytes": v[1] / max(v[2], 1), "launches_per_round": v[2] / n,
                                  "gbs": (v[1] / v[0] / 1e3) if v[0] else 0} for k, v in glin_by.items()},
           "experts_groups": {k: {"us_per_round": v[0] / n, "bytes": v[1] / max(v[2], 1), "launches_per_round": v[2] / n,
                                  "gbs": (v[1] / v[0] / 1e3) if v[0] else 0} for k, v in xp_by.items()},
           "unmatched_glinear": {str(k): v for k, v in unmatched.items()},
           "walls": [r["wall"] for r in per_round], "prompts": prompts}
    if sclk is not None:
        res["sclk_mhz"] = {k: {"n": len(v), "median": statistics.median(v), "mean": sum(v) / len(v)} for k, v in sclk_by.items() if v}
    return res


def load_sclk(path):
    ts, vs = [], []
    for line in open(path):
        if not line.startswith("{"):
            continue
        x = json.loads(line)
        ts.append(x["t"]); vs.append(x["sclk"])
    return ts, vs


def sclk_at(sclk, s, e):
    ts, vs = sclk
    i = bisect.bisect_left(ts, s)
    j = bisect.bisect_right(ts, e)
    if j > i:
        return sum(vs[i:j]) / (j - i)
    return None


def pair_sites(rep):
    out = {}
    for key, a in rep["rank0"]["sites"].items():
        b = rep["rank1"]["sites"].get(key)
        if b:
            out[key] = {"wait0": a["wait_mean"], "wait1": b["wait_mean"], "coll0": a["mean"], "coll1": b["mean"],
                        "p90_0": a["p90"], "p90_1": b["p90"]}
    return out


def render(rep):
    L = []
    for rk in ("rank0", "rank1"):
        if rk not in rep:
            continue
        r = rep[rk]
        m = r["mean_us"]
        L.append("== %s: %d rounds" % (rk, r["rounds"]))
        wall = m.get("wall", 0)
        def row(label, us, extra=""):
            L.append("  %-34s %9.1f us  %5.1f%%  %s" % (label, us, 100 * us / wall if wall else 0, extra))
        gb = m.get("glinear_bytes", 0); xbt = m.get("experts_bytes", 0)
        row("round wall (token period)", wall, "-> %.2f rounds/s" % (1e6 / wall if wall else 0))
        row("GPU busy (union of kernels)", m.get("busy", 0))
        row("experts (grouped_cp)", m.get("experts", 0), "%.1f MB, %.0f GB/s, ideal %.0f us, excess %.0f us" % (
            xbt / 1e6, xbt / m["experts"] / 1e3 if m.get("experts") else 0, xbt / BW * 1e6, m.get("experts", 0) - xbt / BW * 1e6))
        row("experts decode_prep", m.get("experts_prep", 0))
        row("dense glinear", m.get("glinear", 0), "%.1f MB, %.0f GB/s, ideal %.0f us, excess %.0f us" % (
            gb / 1e6, gb / m["glinear"] / 1e3 if m.get("glinear") else 0, gb / BW * 1e6, m.get("glinear", 0) - gb / BW * 1e6))
        row("router (_rowmm_parts)", m.get("router", 0), "%.1f MB" % (m.get("router_bytes", 0) / 1e6))
        row("small kernels", m.get("small", 0))
        row("copies (rocclr)", m.get("copy", 0))
        row("gather stage", m.get("stage", 0))
        row("gather collect", m.get("collect", 0))
        row("launch gaps < 20 us", m.get("gap_launch", 0) - m.get("draft_gap_launch", 0))
        row("host gaps >= 20 us in round", m.get("gap_host", 0) - m.get("draft_gap_host", 0))
        row("drafter (after head gather)", m.get("draft_total", 0) + m.get("draft_gap_launch", 0) + m.get("draft_gap_host", 0),
            " ".join("%s %.0f" % (k[6:], v) for k, v in sorted(m.items()) if k.startswith("draft_") and k not in ("draft_total",) and v))
        row("idle to next round", m.get("idle_after", 0))
        acc = sum(m.get(k, 0) for k in ("experts", "experts_prep", "glinear", "router", "small", "copy", "stage", "collect", "gap_launch", "gap_host", "draft_total", "idle_after"))
        row("(sum of the above)", acc)
        L.append("  dense groups (per round): " + "; ".join("%s %.0f us %.1f MB/launch x%.1f %.0f GB/s" % (k, v["us_per_round"], v["bytes"] / 1e6, v["launches_per_round"], v["gbs"]) for k, v in sorted(r["glinear_groups"].items(), key=lambda kv: -kv[1]["us_per_round"])))
        L.append("  experts (per round): " + "; ".join("%s %.0f us %.1f MB x%.1f %.0f GB/s" % (k, v["us_per_round"], v["bytes"] / 1e6, v["launches_per_round"], v["gbs"]) for k, v in sorted(r["experts_groups"].items())))
        if r["unmatched_glinear"]:
            L.append("  UNMATCHED glinear launches: %s" % r["unmatched_glinear"])
        if "sclk_mhz" in r:
            L.append("  sclk MHz by kernel kind: " + "; ".join("%s %.0f (n %d)" % (k, v["median"], v["n"]) for k, v in sorted(r["sclk_mhz"].items())))
        for p in r.get("prompts", []):
            L.append("  prompt chunk %d rows: span %.0f us; %s" % (p["rows"], p["span_us"], ", ".join("%s %.0f" % kv for kv in sorted(p["by_kind_us"].items(), key=lambda kv: -kv[1]))))
        w = sorted(r["walls"])
        if w:
            L.append("  round wall us: p10 %.0f median %.0f p90 %.0f" % (w[len(w) // 10], w[len(w) // 2], w[9 * len(w) // 10]))
    if "pair" in rep:
        L.append("== gather sites: mean collect us (rank0 / rank1), wait above the site's p5 floor")
        tot0 = tot1 = 0
        for k, v in sorted(rep["pair"].items(), key=lambda kv: sort_key(kv[0])):
            tot0 += v["wait0"]; tot1 += v["wait1"]
            L.append("  %-12s collect %7.1f / %7.1f   wait %7.1f / %7.1f   p90 %7.1f / %7.1f" % (k, v["coll0"], v["coll1"], v["wait0"], v["wait1"], v["p90_0"], v["p90_1"]))
        L.append("  total wait per round: rank0 %.0f us, rank1 %.0f us" % (tot0, tot1))
    return "\n".join(L)


def sort_key(k):
    m = re.match(r"L(\d+)\.(\w+)", k)
    if m:
        return (0, int(m.group(1)), {"engram": 0, "attn": 1, "moe": 2}.get(m.group(2), 3))
    return (1 if k.startswith("head") else 2, 0, k)


if __name__ == "__main__":
    main()
