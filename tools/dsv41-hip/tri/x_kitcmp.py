"""the component kit (~/tfp/kit/tensorfold-strix def87879, zig/kernels/hip/dsv41/model_math.hip: five scalar
baseline kernels) against the Triton kernels that do the same work, on the same inputs: index scores (decode 1 row x
32768 keys, prompt 2048 rows x 8192 keys), top-k 512 (of 4096), mHC mixes + finish (1, 6, 2048 rows) and the posted
streams (1, 6, 2048 rows). Device times come from rocprofv3's kernel trace (run this under it); results are also
compared with each other (the kit's outputs against the Triton outputs, both checked against fp64 elsewhere)."""

import ctypes as Cc
import json
import sys

import numpy as np

import common as C
import hiprt as H
import refs as R
from hiprt import Buf, P

L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
rng = np.random.default_rng(12)
C.log_open("x_kitcmp")
MOD = H.Module((C.ROOT / "kitcmp" / "model_math.hsaco").read_bytes())
U64, U32, F32 = Cc.c_uint64, Cc.c_uint32, Cc.c_float


def fields(names, kind):
    return [(n, kind) for n in names.split()]


class Index(Cc.Structure):
    _fields_ = fields("q keys scales weights vis base candidates scores order_keys status key_rows candidate_stride", U64) + \
        fields("abi rows n heads dim packed candidate_block reserved", U32)


class Topk(Cc.Structure):
    _fields_ = fields("keys vis out scratch status ldkeys ldout ldscratch", U64) + fields("abi rows n k", U32)


class HcPre(Cc.Structure):
    _fields_ = fields("x fn scale base pre_in norm_w out pre_out post comb part trace status", U64) + \
        fields("abi rows dim iters", U32) + fields("eps hc_eps", F32) + fields("reserved0 reserved1", U32)


class HcPost(Cc.Structure):
    _fields_ = fields("g x out post comb status rank_stride", U64) + fields("abi rows dim world", U32)


assert Cc.sizeof(Index) == 128 and Cc.sizeof(Topk) == 80 and Cc.sizeof(HcPre) == 136 and Cc.sizeof(HcPost) == 72


def kit(name, grid, block, struct, reps=10):
    f = MOD.function(name)
    for _ in range(reps):
        H.launch(f, grid, (block, 1, 1), 0, [struct])
    H.sync()


def k_of(fn, pred):
    for c in L[fn]:
        k = {kk: (v if not isinstance(v, dict) else float(v["float"])) for kk, v in c["consts"].items()}
        if pred(c, k):
            return k
    raise KeyError(fn)


def mark(tag):
    """A marker launch (rows tag) so the trace can be split by case: a 1-thread _tap-free memset is not traced, so
    record the case order in the log and rely on run order."""
    C.record(kernel="x_kitcmp", case=tag)


def index_cases():
    IH, ID = 32, 128
    for rows, n in ((1, 32768), (2048, 8192)):
        raw = rng.standard_normal((n, ID))
        _, codes, sc = R.fp4_quant_rows(raw, 32, False)
        codes = R.fp4_pack(codes)
        q, w = C.rand_bf16(rng, (rows, IH, ID), 0.5), C.rand_bf16(rng, (rows, IH), 0.2)
        vis = np.full(rows, n, np.int64)
        qb, wb, kb, sb, vb = Buf.of(q), Buf.of(w), Buf.of(codes), Buf.of(sc), Buf.of(vis)
        keys_kit, keys_tri, st = Buf(rows * n * 8), Buf(rows * n * 8), Buf.of(np.zeros(1, np.uint32))
        mark(f"index_score rows={rows} n={n}")
        a = Index(q=qb.ptr, keys=kb.ptr, scales=sb.ptr, weights=wb.ptr, vis=vb.ptr, base=0, candidates=0, scores=0,
                  order_keys=keys_kit.ptr, status=st.ptr, key_rows=n, candidate_stride=0, abi=1, rows=rows, n=n,
                  heads=IH, dim=ID, packed=1, candidate_block=0, reserved=0)
        kit("tf_dsv41_index_score", ((rows * n + 255) // 256, 1, 1), 256, a, reps=3 if rows > 1 else 10)
        nt = n // 64
        tm = Buf(rows * nt * 8)
        if rows == 1:
            k = k_of("_index_score", lambda c, k: k["KEYS"] and k["HAS_TMAX"] and not k["HAS_BASE"] and "n" not in k)
            args = dict(Q=P(qb, "*bf16"), K=P(kb, "*u8"), KS=P(sb, "*u8"), Wt=P(wb, "*bf16"), VIS=P(vb, "*i64"),
                        OUT=P(keys_tri, "*i64"), n=n, KB=P(vb, "*i64"), CAND=P(vb, "*i64"), cs=0, TMAX=P(tm, "*i64"), nt=nt)
            for _ in range(10):
                C.KIT.run("_index_score", (rows, nt, 1), args, k)
        else:
            k = k_of("_index_score_rows", lambda c, k: k["KEYS"] and k["HAS_TMAX"] and "nt" not in k)
            args = dict(Q=P(qb, "*bf16"), K=P(kb, "*u8"), KS=P(sb, "*u8"), Wt=P(wb, "*bf16"), VIS=P(vb, "*i64"),
                        OUT=P(keys_tri, "*i64"), n=n, rows=rows, TMAX=P(tm, "*i64"), nt=nt)
            for _ in range(3):
                C.KIT.run("_index_score_rows", (rows, nt, 1), args, k)
        H.sync()
        a_ = keys_kit.download((rows, n), np.int64)
        b_ = keys_tri.download((rows, n), np.int64)
        sa = (a_ >> 32).astype(np.int32)
        sb_ = (b_ >> 32).astype(np.int32)
        fa = np.where(sa < 0, sa ^ 0x7FFFFFFF, sa).view(np.float32).astype(np.float64)
        fb = np.where(sb_ < 0, sb_ ^ 0x7FFFFFFF, sb_).view(np.float32).astype(np.float64)
        C.record(kernel="index_score kit vs triton", case=f"rows={rows} n={n}", status=int(st.download(1, np.uint32)[0]),
                 max_rel_diff=float(np.max(np.abs(fa - fb) / np.maximum(np.abs(fb), 1e-6))), bits_equal=float(np.mean(a_ == b_)))


def topk_cases():
    for rows, n in ((1, 4096), (6, 4096), (2048, 4096)):
        s = rng.standard_normal((rows, n)).astype(np.float32)
        bits = s.view(np.int32).astype(np.int64)
        keys = (np.where(bits < 0, bits ^ 0x7FFFFFFF, bits) << 32) | (0xFFFFFFFF - np.arange(n, dtype=np.int64))
        kb, vb = Buf.of(keys), Buf.of(np.full(rows, n, np.int64))
        out_k, out_t, scr, st = Buf(rows * 512 * 8), Buf(rows * 512 * 8), Buf(rows * 512 * 8), Buf.of(np.zeros(1, np.uint32))
        mark(f"topk rows={rows} n={n}")
        a = Topk(keys=kb.ptr, vis=vb.ptr, out=out_k.ptr, scratch=scr.ptr, status=st.ptr, ldkeys=n, ldout=512, ldscratch=512,
                 abi=1, rows=rows, n=n, k=512)
        kit("tf_dsv41_index_topk", (rows, 1, 1), 256, a, reps=3 if rows > 6 else 10)
        k = k_of("_topk_sel", lambda c, k: k["N"] == n and k["KK"] == 512)
        for _ in range(3 if rows > 6 else 10):
            C.KIT.run("_topk_sel", (rows, 1, 1), dict(KEYS=P(kb, "*i64"), VIS=P(vb, "*i64"), OUT=P(out_t, "*i64")), k)
        H.sync()
        C.record(kernel="index_topk kit vs triton", case=f"rows={rows} n={n}", status=int(st.download(1, np.uint32)[0]),
                 equal=bool(np.array_equal(out_k.download((rows, 512), np.int64), out_t.download((rows, 512), np.int64))))


def hc_cases():
    D, HC, NB, WIDE = 5120, 4, 40, 20480
    fn = (rng.standard_normal((24, WIDE)) * 0.01).astype(np.float32)
    sc, base, nw = np.array([0.8, 1.1, 0.6], np.float32), (rng.standard_normal(24) * 0.5).astype(np.float32), C.rand_bf16(rng, (D,), 0.5)
    fnb, scb, bb, nwb = Buf.of(fn), Buf.of(sc), Buf.of(base), Buf.of(nw)
    for rows in (1, 6, 2048):
        x = C.rand_bf16(rng, (rows, HC, D))
        pre_in = (rng.random((rows, HC)) + 1e-6).astype(np.float32)
        xb, pib = Buf.of(x), Buf.of(pre_in)
        o = {n: Buf(rows * s) for n, s in (("out", D * 2), ("pre", 16), ("post", 16), ("comb", 64), ("part", 100))}
        st = Buf.of(np.zeros(1, np.uint32))
        mark(f"hc pre rows={rows}")
        a = HcPre(x=xb.ptr, fn=fnb.ptr, scale=scb.ptr, base=bb.ptr, pre_in=pib.ptr, norm_w=nwb.ptr, out=o["out"].ptr,
                  pre_out=o["pre"].ptr, post=o["post"].ptr, comb=o["comb"].ptr, part=o["part"].ptr, trace=0, status=st.ptr,
                  abi=1, rows=rows, dim=D, iters=20, eps=1e-20, hc_eps=1e-6, reserved0=0, reserved1=0)
        reps = 3 if rows > 6 else 10
        kit("tf_dsv41_hc_mix", (rows * 25, 1, 1), 256, a, reps)
        kit("tf_dsv41_hc_finish", (rows, 1, 1), 256, a, reps)
        # Triton: prompt hc_pre (_hc_partial + _hc_finish) and the decode window's split finish
        part = Buf(rows * NB * 32 * 4)
        t_out = {n: Buf(rows * s) for n, s in (("out", D * 2), ("pre", 16), ("post", 16), ("comb", 64))}
        fa = dict(X=P(xb, "*bf16"), PART=P(part, "*fp32"), BASE=P(bb, "*fp32"), SCALE=P(scb, "*fp32"), PRE_IN=P(pib, "*fp32"),
                  NW=P(nwb, "*bf16"), OUT=P(t_out["out"], "*bf16"), PRE_OUT=P(t_out["pre"], "*fp32"),
                  POST=P(t_out["post"], "*fp32"), COMB=P(t_out["comb"], "*fp32"), eps=1e-20, hc_eps=1e-6)
        for _ in range(reps):
            C.KIT.run("_hc_partial", (rows, NB, 1), dict(X=P(xb, "*bf16"), FN=P(fnb, "*fp32"), PART=P(part, "*fp32")),
                      k_of("_hc_partial", lambda c, k: True))
        for _ in range(reps):
            C.KIT.run("_hc_finish", (rows, 1, 1), fa, k_of("_hc_finish", lambda c, k: True))
        H.sync()
        diff = np.abs(C.from_bf16_bits(o["out"].download((rows, D), np.uint16)) - C.from_bf16_bits(t_out["out"].download((rows, D), np.uint16)))
        C.record(kernel="hc_pre kit vs triton", case=f"rows={rows}", status=int(st.download(1, np.uint32)[0]),
                 out_max_abs_diff=float(diff.max()),
                 comb_max_abs_diff=float(np.abs(o["comb"].download((rows, 16), np.float32) - t_out["comb"].download((rows, 16), np.float32)).max()))
        # posted streams
        g = (rng.standard_normal((2, rows, D))).astype(np.float32)
        post = (2 * rng.random((rows, HC))).astype(np.float32)
        comb = rng.random((rows, HC, HC)).astype(np.float32)
        gb, pb, cb = Buf.of(g), Buf.of(post), Buf.of(comb)
        ok_, ot_ = Buf(rows * HC * D * 2), Buf(rows * HC * D * 2)
        mark(f"hc post rows={rows}")
        a = HcPost(g=gb.ptr, x=xb.ptr, out=ok_.ptr, post=pb.ptr, comb=cb.ptr, status=st.ptr, rank_stride=rows * D, abi=1,
                   rows=rows, dim=D, world=2)
        kit("tf_dsv41_hc_post", ((rows * D + 255) // 256, 1, 1), 256, a, reps)
        for _ in range(reps):
            C.KIT.run("_hc_post", (rows, D // 1024, 1), dict(G=P(gb, "*fp32"), RS=rows * D, X=P(xb, "*bf16"), XOUT=P(ot_, "*bf16"),
                      POST=P(pb, "*fp32"), COMB=P(cb, "*fp32")), k_of("_hc_post", lambda c, k: True))
        H.sync()
        C.record(kernel="hc_post kit vs triton", case=f"rows={rows}", status=int(st.download(1, np.uint32)[0]),
                 bits_equal=float(np.mean(ok_.download((rows, HC, D), np.uint16) == ot_.download((rows, HC, D), np.uint16))))


if __name__ == "__main__":
    index_cases()
    topk_cases()
    hc_cases()
    C.done()
