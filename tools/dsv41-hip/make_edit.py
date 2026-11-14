#!/usr/bin/env python3
"""make_edit.py OUT: the edit file (serve.conf EDIT, engine TF_DS_EDIT) that serves an abliterated DeepSeek V4.1 Flash
fine-tune on the EXL3 pack it was made from, at nearly that pack's speed.

dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED (89af9bf1) is the LibertAIDAI EXL3 2.9bpw pack (c178a012)
with attn.wo_b and ffn.shared_experts.w2 of blocks 10-36 replaced by dense BF16 weights. Each of those 54 weights is,
to its BF16 rounding, W' = diag(g) (I - alpha r r^T) W: W the EXL3 pack's own decoded weight, r one unit direction of
the 5120 outputs (its own for each projection), alpha the block's strength (3 at block 10, 6 at block 20, 1 at block
36), g the scale that gives each output row back its length. So the engine keeps the EXL3 weights and edits the two
projections' outputs (dense.hip's edit); the dense copies read 3.3x the EXL3 bytes every token.

This reads only those tensors: the 54 BF16 weights of the fine-tune and the same projections' EXL3 parts of the pack
(~3.8 GB, by HTTP ranges from Hugging Face, or from local folders), decodes the EXL3 (mul1 codebook, as
docs/recipes/exl3.md writes it down), fits r, alpha and g, and refuses a projection whose fit is not within 1.5x of
the BF16 rounding floor (a weight that is not such an edit), a block with only one of the two dense, and any other
weight the fine-tune stores dense where the pack has EXL3 parts. The file's metadata names the base pack, which
serve_lib.sh checks against the bundle's. It prints each projection's fit and the file's sha256 (serve.conf
EDIT_SHA256). numpy is the one dependency. --check OUT reads a written file back and prints its sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request

import numpy as np

FINETUNE = "dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED@89af9bf193575b7eeeb4f1dd85463ec24ea5bb4c"
BASE = "LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw@c178a01284fc5c0e859d8668cffaac81ca8f4042"
PROJ = ("attn.wo_b", "ffn.shared_experts.w2")
FLOOR_RATIO = 1.5  # a projection's fit leftover may be at most this many times its BF16 rounding floor


class Source:
    """A checkpoint's tensors by name: a local folder, or REPO@REV on Hugging Face read by HTTP ranges."""

    def __init__(self, spec):
        self.local = os.path.isdir(spec)
        if self.local:
            self.root = spec
        else:
            repo, _, rev = spec.partition("@")
            if not rev:
                sys.exit(f"{spec}: name a revision (REPO@COMMIT)")
            self.root = f"https://huggingface.co/{repo}/resolve/{rev}"
        self.index = json.loads(self.read("model.safetensors.index.json"))["weight_map"]
        self.headers = {}

    def read(self, name, start=None, n=None):
        if self.local:
            with open(os.path.join(self.root, name), "rb") as f:
                if start is not None:
                    f.seek(start)
                return f.read() if n is None else f.read(n)
        req = urllib.request.Request(f"{self.root}/{name}")
        if start is not None:
            req.add_header("Range", f"bytes={start}-{start + n - 1}")
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if n is not None and len(data) != n:
            sys.exit(f"{self.root}/{name}: asked for {n} bytes at {start}, got {len(data)}")
        return data

    def header(self, shard):
        if shard not in self.headers:
            n = struct.unpack("<Q", self.read(shard, 0, 8))[0]
            self.headers[shard] = (8 + n, json.loads(self.read(shard, 8, n)))
        return self.headers[shard]

    def has(self, name):
        return name in self.index

    def tensor(self, name):
        shard = self.index[name]
        base, h = self.header(shard)
        t = h[name]
        a, b = t["data_offsets"]
        return t["dtype"], t["shape"], self.read(shard, base + a, b - a) if b > a else b""


def bf16(raw, shape):
    return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)


def bf16_round(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16).view(np.float32).astype(np.float64)


def hadamard(n=128):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


H128 = hadamard()
P = np.arange(256)
ROW = 2 * ((P // 8) % 4) + ((P % 8) & 1) + 8 * (((P % 8) >> 1) & 1)  # value p's place in its 16x16 tile
COL = (P // 8) // 4 + 8 * ((P % 8) >> 2)
C1 = float(np.uint16(0x1EEE).view(np.float16))
C2 = float(np.uint16(0xC931).view(np.float16))


def mul1(s):
    """The mul1 codebook: 16-bit states to fp16 (one fused multiply-add, rounded once)."""
    x = (s.astype(np.uint64) * 0x83DCD12D) & 0xFFFFFFFF
    h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
    return (h.astype(np.float64) * C1 + C2).astype(np.float16)  # exact in f64, then one rounding


def exl3_weight(trellis, suh, svh):
    """W [N, K] (float64) of an EXL3 linear: diag(suh) H_K W_q H_N diag(svh), transposed to the torch layout."""
    kt, nt, words = trellis.shape
    k2, total = words // 8, 16 * words  # half-bits a value, the tile's stream bits
    end = (P + 1) * (k2 // 2) if k2 % 2 == 0 else ((P + 1) * k2 - (P + 1) % 2) // 2  # bits = KA + 1/2: KA, KA + 1, ...
    idx = end[:, None] - 16 + np.arange(16)[None, :]
    idx %= total  # the 16 stream bits that end at value p's end, first bit most significant
    wq = np.empty((kt * 16, nt * 16), np.float16)
    w16 = trellis.view(np.uint16).astype(np.uint32)
    for k in range(kt):
        w32 = w16[k, :, 0::2] | (w16[k, :, 1::2] << 16)
        b = ((w32[:, :, None] >> (31 - np.arange(32))[None, None, :]) & 1).reshape(nt, total).astype(np.uint32)
        s = (b[:, idx] << (15 - np.arange(16))[None, None, :]).sum(-1)
        tile = np.empty((nt, 16, 16), np.float16)
        tile[:, ROW, COL] = mul1(s)
        wq[k * 16:(k + 1) * 16] = tile.transpose(1, 0, 2).reshape(16, nt * 16)
    w = wq.astype(np.float64)
    w = (w.reshape(kt * 16 // 128, 128, -1).transpose(0, 2, 1) @ H128).transpose(0, 2, 1).reshape(kt * 16, -1)
    w = (w.reshape(kt * 16, -1, 128) @ H128).reshape(kt * 16, -1)
    w = suh.astype(np.float64)[:, None] * w * svh.astype(np.float64)[None, :]
    return w.T


def top(z, u, steps=8):
    """The top singular triple of z by power iterations from u (the edit's gap is two orders of magnitude)."""
    for _ in range(steps):
        v = z.T @ u
        v /= np.linalg.norm(v)
        u = z @ v
        s = np.linalg.norm(u)
        u /= s
    return u, s, v


def fit(wb, wu):
    """r, alpha, g of wu ~ diag(g) (I - alpha r r^T) wb, and the fit's leftover and BF16 floor (relative to |wb|)."""
    nb = np.linalg.norm(wb, axis=1)
    r, s, v = top(wu - wb, np.ones(wb.shape[0]) / np.sqrt(wb.shape[0]))
    for _ in range(4):  # alternately: the rows' scale, then the rank-1 edit of the unscaled rows
        a = wb.T @ r
        alpha = -s * np.sign(v @ a) / np.linalg.norm(a)
        g = nb / np.linalg.norm(wb - alpha * np.outer(r, a), axis=1)
        r, s, v = top(wu / g[:, None] - wb, r)
    a = wb.T @ r
    alpha = -s * np.sign(v @ a) / np.linalg.norm(a)
    m = wb - alpha * np.outer(r, a)
    g = nb / np.linalg.norm(m, axis=1)
    m *= g[:, None]
    scale = np.linalg.norm(wb)
    return r, alpha, g, np.linalg.norm(wu - m) / scale, np.linalg.norm(bf16_round(m) - m) / scale


def edited_blocks(ft, base):
    """The blocks whose two projections the fine-tune stores dense where the base has EXL3 parts; any other such weight
    (one projection of a block alone, another projection, an MTP block) stops the tool: the edit would leave it out."""
    names = (n[:-len(".weight")] for n in ft.index if n.endswith(".weight"))
    exl3 = lambda src, n: src.has(n + ".trellis") and not src.has(n + ".weight")  # noqa: E731
    dense = sorted(n for n in names if not ft.has(n + ".trellis") and exl3(base, n))
    by_block, other = {}, []
    for n in dense:
        parts = n.split(".")
        if len(parts) > 2 and parts[0] == "layers" and parts[1].isdigit() and ".".join(parts[2:]) in PROJ:
            by_block.setdefault(int(parts[1]), set()).add(".".join(parts[2:]))
        else:
            other.append(n)
    other += [f"layers.{i} ({' '.join(sorted(ps))} only)" for i, ps in by_block.items() if len(ps) != len(PROJ)]
    if other:
        sys.exit(f"the fine-tune stores weights dense that this edit cannot carry: {', '.join(other[:8])}")
    return sorted(by_block)


def write(path, tensors, finetune, base):
    hdr, off, blobs = {}, 0, []
    for name, arr in tensors:
        raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        hdr[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    hdr["__metadata__"] = {"format": "strixfold-edit/1", "finetune": finetune, "base": base}
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path + ".tmp", "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--finetune", default=FINETUNE, help="the abliterated pack: a folder or REPO@COMMIT")
    ap.add_argument("--base", default=BASE, help="the EXL3 pack it was made from: a folder or REPO@COMMIT")
    ap.add_argument("--base-id", help="with a local --base: the pack it is, REPO@COMMIT (checked against the bundle)")
    ap.add_argument("--blocks", help="only these blocks (comma separated; a partial file, for a quick look)")
    ap.add_argument("--check", action="store_true", help="read OUT back: its tensors and sha256")
    o = ap.parse_args()
    if o.check:
        with open(o.out, "rb") as f:
            raw = f.read()
        n = struct.unpack("<Q", raw[:8])[0]
        hdr = json.loads(raw[8:8 + n])
        names = [k for k in hdr if k != "__metadata__"]
        print(f"{len(names)} tensors, {len(names) // 6} blocks; sha256 {hashlib.sha256(raw).hexdigest()}")
        return
    ft, base = Source(o.finetune), Source(o.base)
    if o.base_id and not base.local and o.base_id != o.base:
        sys.exit(f"--base-id {o.base_id} names another pack than --base {o.base}")
    base_id = o.base_id or (None if base.local else o.base)
    if not base_id or "@" not in base_id:
        sys.exit("a local --base needs --base-id REPO@COMMIT: the pack it is (serve_lib.sh checks the bundle's)")
    blocks = edited_blocks(ft, base)
    if o.blocks:
        blocks = [i for i in blocks if i in {int(x) for x in o.blocks.split(",")}]
    if not blocks:
        sys.exit(f"{o.finetune}: no block has dense {' and '.join(PROJ)} weights")
    print(f"blocks {blocks[0]}-{blocks[-1]} ({len(blocks)}): fitting {2 * len(blocks)} projections", flush=True)
    tensors, bad = [], []
    for i in blocks:
        for p in PROJ:
            t0 = time.time()
            n = f"layers.{i}.{p}"
            if not base.has(f"{n}.mul1"):
                sys.exit(f"{n}: the base's EXL3 parts have no mul1 marker (this tool decodes mul1 only)")
            dt, shape, raw = ft.tensor(f"{n}.weight")
            if dt != "BF16":
                sys.exit(f"{n}.weight is {dt}, not BF16")
            wu = bf16(raw, shape).astype(np.float64)
            _, tshape, traw = base.tensor(f"{n}.trellis")
            suh = np.frombuffer(base.tensor(f"{n}.suh")[2], np.float16)
            svh = np.frombuffer(base.tensor(f"{n}.svh")[2], np.float16)
            wb = exl3_weight(np.frombuffer(traw, np.int16).reshape(tshape), suh, svh)
            if wb.shape != wu.shape:
                sys.exit(f"{n}: the base decodes to {wb.shape}, the fine-tune's weight is {wu.shape}")
            r, alpha, g, left, floor = fit(wb, wu)
            ok = left <= FLOOR_RATIO * floor
            print(f"{n}: alpha {alpha:.4f}, fit leftover {left:.2e} (BF16 floor {floor:.2e})"
                  f"{'' if ok else '  NOT AN EDIT'}  {time.time() - t0:.0f}s", flush=True)
            if not ok:
                bad.append(n)
            tensors += [(f"{n}.edit.r", r), (f"{n}.edit.g", g), (f"{n}.edit.alpha", np.array([alpha]))]
    if bad:
        sys.exit(f"{len(bad)} projection(s) are not a rank-1 edit of the base: {', '.join(bad)}; no file written")
    write(o.out, tensors, o.finetune if not ft.local else f"local:{os.path.abspath(o.finetune)}", base_id)
    with open(o.out, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"wrote {o.out}: {len(tensors)} tensors; sha256 {digest} (serve.conf EDIT_SHA256)")


if __name__ == "__main__":
    main()
