"""The HIP heads RoPE (aux_rope.hip, tf_ds_aux_rope_heads) bit for bit against the kit's Triton _rope_heads (every
recorded H / HD / INV variant, 1-2048 rows) and within the rounding interval of the float64 rotation."""

import ctypes
import json
import os
import sys

import numpy as np

import common as C
import hiprt as H
import refs as R
from hiprt import Buf, P

C.log_open("t_hiprope")
L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
OBJ = os.environ.get("TRI_ROPE_HSACO", str(C.ROOT / "build" / "aux_rope.hsaco"))
FN = H.Module(open(OBJ, "rb").read()).function("tf_ds_aux_rope_heads")
rng = np.random.default_rng(64)
RD, NPOS = 64, 70000


class RopeArgs(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in "x cos sin pos".split()] + \
               [(n, ctypes.c_int32) for n in "rows h hd rd flags pad0".split()]


assert ctypes.sizeof(RopeArgs) == 56


def hip_rope(x, cos, sin, pos, inv):
    rows, h, hd = x.shape
    xb = Buf.of(x)
    keep = [Buf.of(cos), Buf.of(sin), Buf.of(pos)]
    a = RopeArgs(x=xb.ptr, cos=keep[0].ptr, sin=keep[1].ptr, pos=keep[2].ptr, rows=rows, h=h, hd=hd, rd=RD,
                 flags=1 if inv else 0)
    n = rows * h * (RD // 8)
    H.launch(FN, ((n + 255) // 256, 1, 1), (256, 1, 1), 0, [a])
    H.sync()
    return xb.download(x.shape, np.uint16)


def main():
    ang = rng.uniform(0, 2 * np.pi, (NPOS, RD // 2))
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    seen = set()
    for c in L["_rope_heads"]:
        k = {kk: v for kk, v in c["consts"].items()}
        key = (k["H"], k["HD"], k["INV"], k["HB"])
        if key in seen:
            continue
        seen.add(key)
        h, hd, inv = k["H"], k["HD"], bool(k["INV"])
        for rows in (1, 17, 2048, 8192):
            x = C.rand_bf16(rng, (rows, h, hd), 2.0)
            pos = rng.integers(0, NPOS, rows).astype(np.int64)
            got = hip_rope(x, cos, sin, pos, inv)
            tb = Buf.of(x)
            C.KIT.run("_rope_heads", (rows, h // k["HB"], 1), dict(X=P(tb, "*bf16"), COS=P(Buf.of(cos), "*fp32"),
                      SIN=P(Buf.of(sin), "*fp32"), POS=P(Buf.of(pos), "*i64")), k)
            H.sync()
            tri = tb.download(x.shape, np.uint16)
            # float64 rotation and its fp32 bound (2 products and one add of the tables' values) before the bf16 rounding
            xs = C.from_bf16_bits(x)[..., hd - RD:].reshape(rows, h, RD // 2, 2)
            cs = cos[pos].astype(np.float64)[:, None, :]
            sn = (-1 if inv else 1) * sin[pos].astype(np.float64)[:, None, :]
            re = xs[..., 0] * cs - xs[..., 1] * sn
            im = xs[..., 0] * sn + xs[..., 1] * cs
            ex = np.stack([re, im], -1).reshape(rows, h, RD)
            bnd = C.gamma(2) * np.stack([np.abs(xs[..., 0] * cs) + np.abs(xs[..., 1] * sn),
                                         np.abs(xs[..., 0] * sn) + np.abs(xs[..., 1] * cs)], -1).reshape(rows, h, RD)
            g = C.from_bf16_bits(got)
            m = C.cmp_rounded(g[..., hd - RD:], ex, R.bf16(ex),
                              C.ulp_bf16, bound=bnd)
            untouched = bool(np.array_equal(got[..., :hd - RD], x[..., :hd - RD]))
            tag = f"H={h} HD={hd} INV={inv} rows={rows}"
            C.record(kernel="aux.rope_heads", case=tag + " vs fp64", **m)
            C.record(kernel="aux.rope_heads", case=tag + " other dims untouched, bits equal Triton _rope_heads",
                     ok=untouched and bool(np.array_equal(got, tri)), untouched=untouched,
                     bits_equal=float(np.mean(got == tri)))
    sys.exit(C.done())


if __name__ == "__main__":
    main()
