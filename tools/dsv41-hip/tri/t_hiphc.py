"""The HIP mHC prompt mixes (aux_hc.hip, tf_ds_aux_hc_mix_pf_d5120) against the float64 references of t_hc (posted
streams within their rounding interval, every block's 24 dots and sum of squares within gamma(512) of the exact
sums), posted (world 1 and 2) and plain; row invariance (a row alone vs inside its chunk); and the Triton _hc_mix_pf
on the same inputs (posted streams bit for bit, mixes reported)."""

import ctypes
import json
import os
import sys

import numpy as np

import common as C
import hiprt as H
import t_hc as TH
from hiprt import Buf, P

C.log_open("t_hiphc")
L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
OBJ = os.environ.get("TRI_HC_HSACO", str(C.ROOT / "build" / "aux_hc.hsaco"))
FN = H.Module(open(OBJ, "rb").read()).function("tf_ds_aux_hc_mix_pf_d5120")
D, HC, NB = 5120, 4, 40
RB = int(os.environ.get("TRI_HC_RB", "32"))             # rows per workgroup (the build's)
rng = np.random.default_rng(777)


class HcArgs(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in "x xo g post comb fn part".split()] + [("rs", ctypes.c_int64)] + \
               [(n, ctypes.c_int32) for n in "rows world flags pad0".split()]


assert ctypes.sizeof(HcArgs) == 80


def hip_mix(x, posted=None):
    """posted = (g [world, rows, D], post, comb) or None (plain); returns PART [rows, NB, 32] and xo bits."""
    rows = x.shape[0]
    part = Buf(rows * NB * 32 * 4)
    part.fill(0)
    xb = Buf.of(x)
    keep = [xb]
    if posted is not None:
        g, post, comb = posted
        xob = Buf(rows * HC * D * 2)
        xob.fill(0x5A)
        gb, pb, cb = Buf.of(g), Buf.of(post), Buf.of(comb)
        keep += [gb, pb, cb]
        a = HcArgs(x=xb.ptr, xo=xob.ptr, g=gb.ptr, post=pb.ptr, comb=cb.ptr, fn=TH.FN_B.ptr, part=part.ptr,
                   rs=rows * D, rows=rows, world=g.shape[0], flags=1)
    else:
        xob = None
        a = HcArgs(x=xb.ptr, xo=0, g=0, post=0, comb=0, fn=TH.FN_B.ptr, part=part.ptr, rs=0, rows=rows, world=1, flags=0)
    H.launch(FN, (((rows + RB - 1) // RB) * 10, 1, 1), (256, 1, 1), 0, [a])
    H.sync()
    p = part.download((rows, NB, 32), np.float32)
    return p, (xob.download((rows, HC, D), np.uint16) if xob is not None else None)


def triton_mix_pf(x, posted):
    c = L["_hc_mix_pf"][1]
    rows = x.shape[0]
    k = TH.consts_of(c)
    g, post, comb = posted
    part, xob = Buf(rows * NB * 32 * 4), Buf(rows * HC * D * 2)
    part.fill(0)
    args = dict(X=P(Buf.of(x), "*bf16"), XO=P(xob, "*bf16"), G=P(Buf.of(g), "*fp32"), RS=rows * D,
                POST=P(Buf.of(post), "*fp32"), COMB=P(Buf.of(comb), "*fp32"), FN=P(TH.FN_B, "*fp32"),
                PART=P(part, "*fp32"), rows=rows)
    C.KIT.run("_hc_mix_pf", (((rows + 7) // 8) * NB, 1, 1), args, k)
    H.sync()
    return part.download((rows, NB, 32), np.float32), xob.download((rows, HC, D), np.uint16)


def main():
    for rows in (1, 17, 33, 512):
        x = TH.streams(rows)
        for world in (1, 2, None):
            tag = f"rows={rows} " + (f"posted world={world}" if world else "plain")
            posted = TH.posted_inputs(rows, world) if world else None
            part, xo = hip_mix(x, posted)
            if world:
                exact, bound = TH.post_ref(posted[0], x, posted[1], posted[2])
                m = C.cmp_rounded(C.from_bf16_bits(xo), exact, TH.R.bf16(exact), C.ulp_bf16, bound=bound)
                C.record(kernel="aux.hc_mix_pf", case=tag + " posted streams", **m)
                TH.check_part("aux.hc_mix_pf", tag, part.astype(np.float64), xo, rows)
            else:
                TH.check_part("aux.hc_mix_pf", tag, part.astype(np.float64), x, rows)
            untouched = bool(np.all(part[:, :, 25:] == 0))
            C.record(kernel="aux.hc_mix_pf", case=tag + " slots 25..31 untouched", ok=untouched)
            # row invariance: a few rows alone
            same = True
            for r in sorted({0, rows // 2, rows - 1}):
                p1 = None if posted is None else (posted[0][:, r:r + 1], posted[1][r:r + 1], posted[2][r:r + 1])
                one, xo1 = hip_mix(x[r:r + 1], p1)
                same &= bool(np.array_equal(one[0].view(np.uint32), part[r].view(np.uint32)))
                if xo is not None:
                    same &= bool(np.array_equal(xo1[0], xo[r]))
            C.record(kernel="aux.hc_mix_pf", case=tag + " row invariance", ok=same)
            if world == 2 and rows > 1:
                tp, txo = triton_mix_pf(x, posted)
                rel = np.abs(part[:, :, :25].astype(np.float64) - tp[:, :, :25]) / np.maximum(np.abs(tp[:, :, :25]), 1e-30)
                C.record(kernel="aux.hc_mix_pf", case=tag + " vs Triton _hc_mix_pf posted streams bits", ok=bool(np.array_equal(txo, xo)))
                C.record(kernel="aux.hc_mix_pf", case=tag + " vs Triton _hc_mix_pf mixes", ok=None,
                         bits_equal=float(np.mean(part[:, :, :25] == tp[:, :, :25])), max_rel_diff=float(rel.max()))
    sys.exit(C.done())


if __name__ == "__main__":
    main()
