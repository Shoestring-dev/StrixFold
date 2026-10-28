#!/usr/bin/env python3
"""mkexp2.py SRC BASE OUT EXP: k-dec2's experiment variants of exl3.hip (timing / invariance tests only, never committed).
wmall: glinear's decode / verify passes (1-16 rows, below EXL3_WIDE_MIN) all on WMMA 16x16x16 tiles, the 1-row pass
       included (BASE's strip_tiles, the pre-a11df7c multi-row strip): one arithmetic for every row count, so rows stay
       invariant to the window's width; what the >8-row regression's fix would cost at 1-8 rows and gain at 9-63.
       grouped_cp (the experts) keeps the row-invariant dot2 path."""
import re, sys

src, base, out, exp = sys.argv[1:5]
s = open(src).read()
b = open(base).read()


def rep(a, c, n=1):
    global s
    assert s.count(a) == n, (a[:80], s.count(a))
    s = s.replace(a, c)


if exp == "wmall":
    m = re.search(r"template <int K2>\n__device__ __forceinline__ void strip_tiles\(const uint32_t\* tiles.*?\n}\n", b, re.S)
    assert m, "base strip_tiles"
    fn = m.group(0).replace("void strip_tiles(", "void strip_tiles_wmma(")
    anchor = "// ==== dense linears:"
    rep(anchor, "// k-dec2 wmall: the pre-a11df7c WMMA strip, every glinear decode pass\n" + fn + "\n" + anchor)
    rep("""        if (R == 1) {
            strip_tiles_1row<K2>(tiles, L.stride_k, kt0, per_warp, xh + (size_t)m0 * L.ldx, stage[wave], lane, acc);
        } else {
            const int ar = m0 + min(lane >> 1, R - 1);
            strip_tiles_rows<K2>(tiles, L.stride_k, kt0, per_warp, xh + (size_t)ar * L.ldx, R, stage[wave],
                                 reinterpret_cast<uint4*>(red + wave * 128), lane, acc);
        }""", """        {
            const int ar = m0 + min(lane & 15, R - 1);
            strip_tiles_wmma<K2>(tiles, L.stride_k, kt0, per_warp, xh + (size_t)ar * L.ldx, stage[wave], lane, acc);
        }""")
else:
    sys.exit("unknown experiment " + exp)
open(out, "w").write(s)
