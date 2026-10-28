"""The HIP sparse attention (aux_attn.hip, tf_ds_aux_sparse_attn_h32_d512) on every recorded _sparse_attn_part case:
split mode merged by the kit's Triton _sparse_attn_merge, one-pass mode directly; against the float64 reference with
t_attn's contract (bound + normalised L2, a key-dropped negative control); row invariance (a row alone vs inside the
window); and agreement with the Triton kernel on the same inputs (reported)."""

import ctypes
import json
import os
import sys

import numpy as np

import common as C
import hiprt as H
import refs as R
import t_attn as TA
from hiprt import Buf, P

L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
C.log_open("t_hipattn")
OBJ = os.environ.get("TRI_ATTN_HSACO", str(C.ROOT / "build" / "aux_attn.hsaco"))
BLOCK = int(os.environ.get("TRI_ATTN_BLOCK", "256"))          # threads a workgroup (registry.json's block)
FN = H.Module(open(OBJ, "rb").read()).function("tf_ds_aux_sparse_attn_h32_d512")
U64, I32, F32 = ctypes.c_uint64, ctypes.c_int32, ctypes.c_float


class AttnArgs(ctypes.Structure):
    _fields_ = [(n, U64) for n in "q wsrc wlo comp csc idx pos pm pl po sink out rbase cbase".split()] + \
               [(n, I32) for n in "ring_size n_idx rows heads win splits flags pad0".split()] + [("scale", F32), ("pad1", I32)]


assert ctypes.sizeof(AttnArgs) == 152


def hip_attn(d, splits, keep):
    """Run the HIP kernel on t_attn.build's inputs d; returns (out bf16 bits [rows, H, HD], parts or None)."""
    rows, Hh, HD = d["rows"], d["Hh"], 512
    final = splits == 1 and d["final"]
    out = Buf(rows * Hh * HD * 2)
    pm, pl, po = Buf(rows * Hh * splits * 4), Buf(rows * Hh * splits * 4), Buf(rows * Hh * splits * HD * 4)
    bufs = dict(q=Buf.of(d["q"]), wsrc=Buf.of(d["wsrc"]), wlo=Buf.of(d["wlo"]), pos=Buf.of(d["pos"]),
                sink=Buf.of(d["sink"]))
    if d["has_comp"]:
        bufs["comp"] = Buf.of(d["codes"])
        bufs["csc"] = Buf.of(d["scales"]) if d["packed"] else bufs["wsrc"]
        bufs["idx"] = Buf.of(d["idx"])
    if d["rbase"] is not None:
        bufs["rbase"] = Buf.of(d["rbase"])
    if d["cbase"] is not None:
        bufs["cbase"] = Buf.of(d["cbase"])
    keep.append((bufs, out, pm, pl, po))
    flags = (1 if d["ring"] else 0) | (2 if d["has_comp"] else 0) | (4 if d["packed"] else 0) | \
        (8 if final else 0) | (16 if d["has_base"] else 0)
    g = lambda k: bufs[k].ptr if k in bufs else 0
    a = AttnArgs(q=g("q"), wsrc=g("wsrc"), wlo=g("wlo"), comp=g("comp"), csc=g("csc"), idx=g("idx"), pos=g("pos"),
                 pm=pm.ptr, pl=pl.ptr, po=po.ptr, sink=g("sink"), out=out.ptr,
                 rbase=g("rbase"), cbase=g("cbase") or g("rbase"),
                 ring_size=int(d["ring_size"]), n_idx=int(d["n_idx"]), rows=rows, heads=Hh, win=128, splits=splits,
                 flags=flags, pad0=0, scale=d["scale"], pad1=0)
    H.launch(FN, (rows, 1, splits), (BLOCK, 1, 1), 0, [a])
    H.sync()
    if final:
        return out, None
    return out, (pm, pl, po)


def merge(d, parts, splits, out):
    mk = TA.consts_of(L["_sparse_attn_merge"][0])
    kk = mk if splits == mk["SPLITS"] else None
    args = dict(PM=P(parts[0], "*fp32"), PL=P(parts[1], "*fp32"), PO=P(parts[2], "*fp32"),
                SINK=P(Buf.of(d["sink"]), "*fp32"), OUT=P(out, "*bf16"))
    if kk is not None:
        C.KIT.run("_sparse_attn_merge", (d["rows"], d["Hh"] // 16, 1), args, kk)
    else:
        C.run_variant(C.extra_variant("_sparse_attn_merge", L["_sparse_attn_merge"][0], SPLITS=splits),
                      (d["rows"], d["Hh"] // 16, 1), args)
    H.sync()


def check(tag, got, d):
    keys = TA.keys_of(d)
    ref, bnd = TA.attn_ref(d["q"], keys, d["sink"].astype(np.float64), d["scale"])
    m = C.cmp_rounded(got, ref, R.bf16(ref), C.ulp_bf16, bound=bnd, max_frac=1.0)
    m["ok"] = m["ok"] and m["nl2"] <= 4e-3
    C.record(kernel="aux.sparse_attn", case=tag, **m)
    drop = [k_[1:] if k_.shape[0] > 1 else k_ for k_ in keys]
    ref2, bnd2 = TA.attn_ref(d["q"], drop, d["sink"].astype(np.float64), d["scale"])
    m2 = C.cmp_rounded(got, ref2, R.bf16(ref2), C.ulp_bf16, bound=bnd2, max_frac=1.0)
    C.record(kernel="aux.sparse_attn", case=tag + " negative control (one key dropped)",
             ok=not (m2["ok"] and m2["nl2"] <= 4e-3), nl2=m2["nl2"])


def sub(d, r):
    """Row r of t_attn.build's inputs alone."""
    e = dict(d)
    e["rows"] = 1
    for k in ("q", "pos"):
        e[k] = d[k][r:r + 1].copy()
    for k in ("rbase", "cbase", "idx"):
        if d.get(k) is not None:
            e[k] = d[k][r:r + 1].copy()
    return e


def main():
    keep = []
    for ci, c in enumerate(L["_sparse_attn_part"]):
        k = TA.consts_of(c)
        rows = min(c["tensors"]["Q"][1][0], 40)
        d = TA.build(c, rows)
        final = bool(k["FINAL"])
        splits = 1 if final else k["SPLITS"]
        out, parts = hip_attn(d, splits, keep)
        if parts is not None:
            merge(d, parts, splits, out)
        got = C.from_bf16_bits(out.download((rows, d["Hh"], 512), np.uint16))
        tag = (f"rec{ci} rows={rows} ring={d['ring']} comp={d['has_comp']} packed={d['packed']} base={d['has_base']} "
               f"final={final} n_idx={d['n_idx']} splits={splits}")
        check(tag, got, d)
        # the Triton kernel on the same inputs: how close the two AMD implementations are
        parts_t = (Buf(rows * d["Hh"] * 8 * 4), Buf(rows * d["Hh"] * 8 * 4), Buf(rows * d["Hh"] * 8 * 512 * 4))
        out_t = TA.run_part(d, c, parts_t)
        if not final:
            merge(d, parts_t, k["SPLITS"], out_t)
        got_t = C.from_bf16_bits(out_t.download((rows, d["Hh"], 512), np.uint16))
        C.record(kernel="aux.sparse_attn", case=tag + " vs Triton", ok=None,
                 nl2_vs_triton=float(np.linalg.norm(got - got_t) / max(np.linalg.norm(got_t), 1e-30)),
                 bits_equal=float(np.mean(got == got_t)))
        if rows > 1 and ci in (2, 4, 9):
            same = True
            for r in (0, rows // 2, rows - 1):
                e = sub(d, r)
                o1, p1 = hip_attn(e, splits, keep)
                if p1 is not None:
                    merge(e, p1, splits, o1)
                same &= bool(np.array_equal(o1.download((1, d["Hh"], 512), np.uint16)[0],
                                            out.download((rows, d["Hh"], 512), np.uint16)[r]))
            C.record(kernel="aux.sparse_attn", case=f"rec{ci} row invariance (rows alone vs in {rows})", ok=same)
    H.sync()
    sys.exit(C.done())


if __name__ == "__main__":
    main()
