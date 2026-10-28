"""The HIP router gate and indexer head weights (aux_gate.hip, tf_ds_aux_gate_logits_d5120 and _n32_d5120) against
the float64 reference: logits within
gamma(5120) of the exact sums (random weights, edge-case fp16 weights, and the checkpoint's own gate weights), row
invariance (a row alone vs inside its chunk), and the routing: the kit's _route on the HIP logits graded by route_check
(numerics.json router.noaux_tc: picks within the band rule of the exact scores, weights within their clause), and
against the engine's current path (hipBLAS sgemm on fp32 copies): rows whose picks differ, all of them inside the band
rule."""

import ctypes
import json
import os
import sys

import numpy as np

import common as C
import hiprt as H
import t_real as TRL                     # checkpoint tensors (its log opens first)
import t_route as TR
from hiprt import Buf, P

C.log_open("t_hipgate")
L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
OBJ = os.environ.get("TRI_GATE_HSACO", str(C.ROOT / "build" / "aux_gate.hsaco"))
MOD = H.Module(open(OBJ, "rb").read())
FN = MOD.function("tf_ds_aux_gate_logits_d5120")             # ne a multiple of 128: 64 x 128 a workgroup, 256 threads
FN32 = MOD.function("tf_ds_aux_gate_logits_n32_d5120")       # ne = 32 (the indexer's head weights): 32 x 32, 128
D, BM, BN = 5120, 64, 128


def build_of(ne):
    """(function, rows per workgroup, outputs per workgroup, threads) of the build for ne outputs."""
    return (FN32, 32, 32, 128) if ne == 32 else (FN, BM, BN, 256)
rng = np.random.default_rng(5150)


class GateArgs(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in "x w out".split()] + [(n, ctypes.c_int32) for n in "rows ne flags pad0".split()]


assert ctypes.sizeof(GateArgs) == 40


def hip_gate(x, w):
    rows, ne = x.shape[0], w.shape[0]
    xb, wb, out = Buf.of(x), Buf.of(w), Buf(rows * ne * 4)
    out.fill(0x5A)
    a = GateArgs(x=xb.ptr, w=wb.ptr, out=out.ptr, rows=rows, ne=ne, flags=0)
    f, bm, bn, nt = build_of(ne)
    H.launch(f, ((rows + bm - 1) // bm, ne // bn, 1), (nt, 1, 1), 0, [a])
    H.sync()
    return out.download((rows, ne), np.float32)


class HipBlas:
    """hipblasSgemm as blas.zig's xwT issues it (column-major C^T = W x^T: transa T, transb N, m = ne, n = rows)."""

    def __init__(self):
        self.lib = ctypes.CDLL("libhipblas.so")
        self.h = ctypes.c_void_p()
        assert self.lib.hipblasCreate(ctypes.byref(self.h)) == 0
        self.lib.hipblasSgemm.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                          ctypes.c_int]

    def xwT(self, xf, wf, out, rows, k, n):
        one, zero = ctypes.c_float(1.0), ctypes.c_float(0.0)
        assert self.lib.hipblasSgemm(self.h, 112, 111, n, rows, k, ctypes.byref(one), ctypes.c_void_p(wf), k,
                                     ctypes.c_void_p(xf), k, ctypes.byref(zero), ctypes.c_void_p(out), n) == 0


def blas_gate(blas, x, w):
    rows, ne = x.shape[0], w.shape[0]
    xf, wf, out = Buf.of(C.from_bf16_bits(x).astype(np.float32)), Buf.of(w.astype(np.float32)), Buf(rows * ne * 4)
    blas.xwT(xf.ptr, wf.ptr, out.ptr, rows, D, ne)
    H.sync()
    return out.download((rows, ne), np.float32)


def ref(x, w):
    xs, ws = C.from_bf16_bits(x), w.astype(np.float64)
    return xs @ ws.T, np.abs(xs) @ np.abs(ws).T


def check(tag, x, w):
    got = hip_gate(x, w)
    ex, ab = ref(x, w)
    m = C.cmp_bound(got.astype(np.float64), ex, C.gamma(D) * ab)
    C.record(kernel="aux.gate_logits", case=tag, **m)
    same = True
    for r in sorted({0, x.shape[0] // 2, x.shape[0] - 1}):
        same &= bool(np.array_equal(hip_gate(x[r:r + 1], w)[0].view(np.uint32), got[r].view(np.uint32)))
    C.record(kernel="aux.gate_logits", case=tag + " row invariance", ok=same)
    return got, ex, ab


def edge_weights(ne):
    """fp16 weights with full 11-bit significands, subnormals, the largest finite values and signed zeros."""
    w = rng.standard_normal((ne, D)).astype(np.float16)
    bits = w.view(np.uint16)
    bits[:, 0::7] |= 0x03FF & rng.integers(0, 1024, bits[:, 0::7].shape).astype(np.uint16)   # full significands
    w = bits.view(np.float16)
    w[:, 1::97] = np.float16(6.0e-8) * rng.integers(1, 1024, w[:, 1::97].shape)               # subnormals
    w[:, 2::389] = np.float16(65504.0) * np.sign(rng.standard_normal(w[:, 2::389].shape))    # the largest values
    w[:, 3::211] = np.float16(-0.0)
    return w


def routing(tag, ne, topk, slots, x, w, bias, blas):
    kr = TRL.k_of("_route", lambda c, k: k["NE"] == ne)
    shared_id = ne

    def route(lg):
        pk, wt = Buf(len(lg) * slots * 4), Buf(len(lg) * slots * 4)
        C.KIT.run("_route", (len(lg), 1, 1), dict(L=P(Buf.of(lg), "*fp32"), BIAS=P(Buf.of(bias), "*fp32"),
                  PICK=P(pk, "*i32"), WTS=P(wt, "*fp32"), scale=1.5, shared_id=shared_id), kr)
        return pk.download((len(lg), slots), np.int32), wt.download((len(lg), slots), np.float32).astype(np.float64)

    got, ex, ab = check(tag, x, w)
    pk, wt = route(got)
    TR.route_check("aux.gate_logits + _route", tag + " routing vs exact", pk, wt, ex, bias, topk, slots, shared_id, 1.5,
                   C.gamma(D) * ab)
    # the engine's current path: hipBLAS on fp32 copies, then the same _route
    bl = blas_gate(blas, x, w)
    mb = C.cmp_bound(bl.astype(np.float64), ex, C.gamma(D) * ab)
    pb, _ = route(bl)
    differ = [r for r in range(len(x)) if not np.array_equal(np.sort(pk[r, :topk]), np.sort(pb[r, :topk]))]
    reordered = [r for r in range(len(x)) if r not in differ and not np.array_equal(pk[r], pb[r])]
    # a differing row is explained when both pick sets pass numerics.json's band rule (route_check's)
    ref_picks, _, _, _ = TR.R.gate(ex, bias, topk, 1.5)
    sc, dsc = TR.score_bound(ex, C.gamma(D) * ab)
    sel = sc + bias.astype(np.float64)
    must, free = TR.selection(sel, 2 * (dsc + TR.U32 * np.abs(sel)), ref_picks, topk)
    unexplained = [r for r in differ if not (TR.set_ok(must[r], free[r], pk[r, :topk], topk)
                                             and TR.set_ok(must[r], free[r], pb[r, :topk], topk))]
    # rows the band rule leaves a choice in (more FREE experts than picks left after MUST)
    wref_ambiguous = int(np.sum(free.sum(1) > topk - must.sum(1)))
    C.record(kernel="aux.gate_logits + _route", case=tag + " picks vs the hipBLAS path", ok=not unexplained,
             rows=len(x), differing_rows=len(differ), reordered_rows=len(reordered), ambiguous_rows=wref_ambiguous,
             unexplained_rows=len(unexplained), hipblas_logits_max_err_over_bound=mb["max_err_over_bound"])


def main():
    for rows, ne in ((1, 384), (17, 384), (64, 128), (100, 384), (512, 384)):
        x = C.rand_bf16(rng, (rows, D), 1.0)
        w = (rng.standard_normal((ne, D)) * 0.02).astype(np.float16)
        check(f"random rows={rows} ne={ne}", x, w)
    x = C.rand_bf16(rng, (70, D), 1.0)
    check("edge-case fp16 weights rows=70 ne=128", x, edge_weights(128))
    check("edge-case fp16 weights rows=70 ne=32", x, edge_weights(32))
    blas = HipBlas()
    # the indexer's head weights (weights_proj, fp16 [32, 5120]; the engine's prompt path: hipBLAS on fp32 copies)
    for layer in ("layers.2", "layers.24"):
        w = TRL.tensor(f"{layer}.attn.indexer.weights_proj.weight")
        for rows in (1, 17, 2048):
            x = C.rand_bf16(rng, (rows, D), 1.0)
            got, ex, ab = check(f"real {layer} weights_proj rows={rows} ne=32", x, w)
            bl = blas_gate(blas, x, w)
            mb = C.cmp_bound(bl.astype(np.float64), ex, C.gamma(D) * ab)
            C.record(kernel="aux.gate_logits", case=f"real {layer} weights_proj rows={rows}: the hipBLAS path's logits",
                     ok=mb["ok"], max_err_over_bound=mb["max_err_over_bound"],
                     max_abs_diff_hip_vs_hipblas=float(np.max(np.abs(got.astype(np.float64) - bl))))
    for layer, ne, topk, slots in (("layers.20", 384, 6, 7), ("layers.2", 384, 6, 7), ("mtp.2", 128, 3, 4)):
        w = TRL.tensor(f"{layer}.ffn.gate.weight")
        bias = TRL.tensor(f"{layer}.ffn.gate.bias").astype(np.float32)
        for rows in (256, 2048):
            routing(f"real {layer} rows={rows} ne={ne}", ne, topk, slots, C.rand_bf16(rng, (rows, D), 1.0), w, bias, blas)
    sys.exit(C.done())


if __name__ == "__main__":
    main()
