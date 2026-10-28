"""The family's native kernels on gfx1151 (kit/gfx1151-native, from make_hip.py) with the launch geometry ops.zig /
exact.zig use, against numpy references: exact where the operation is exact (casts, copies, gathers, selections,
masks), fp32 bounds where it computes; plus the cuBLAS replacement (hipBLAS sgemm in torch's call shape)."""

import ctypes
import os
import sys

import numpy as np

import common as C
import hiprt as H
import refs as R
from hiprt import Buf

rng = np.random.default_rng(31337)
C.log_open("t_native")
NAT = C.ROOT / "kit" / "gfx1151-native"
MODS = {m: H.Module((NAT / f"{m}.hsaco").read_bytes()) for m in ("dsv41_ops", "dsv41_torch", "pointwise", "movement")}
# w-tri's 12 kernels come from the registered aux.hip object (zig/kernels/hip/dsv41/aux.hip, built with the kit flags:
# TRI_AUX_HSACO); the glue kernels (w-arch's ids) from the generated port
AUX = os.environ.get("TRI_AUX_HSACO", str(C.ROOT / "build" / "aux.hsaco"))
MODS["aux"] = H.Module(open(AUX, "rb").read())
AUX_OPS = {"tf_ds_topk_i64_kernel", "tf_ds_topk_indices_kernel", "tf_ds_iota_vis_kernel", "tf_ds_cand_fast_kernel",
           "tf_ds_block_max_kernel", "tf_ds_pool_pick_kernel", "tf_ds_apply_pool_kernel", "tf_ds_rms_norm_kernel",
           "tf_ds_compress2_kernel", "tf_ds_rope_kernel", "tf_ds_bf16_scale_kernel", "tf_ds_hc_mean4_kernel"}
u64, i64, i32, u32, f32 = ctypes.c_uint64, ctypes.c_int64, ctypes.c_int32, ctypes.c_uint32, ctypes.c_float


def fn(mod, name):
    if name in AUX_OPS:
        return MODS["aux"].function("tf_ds_aux_" + name[len("tf_ds_"):-len("_kernel")])
    return MODS[mod].function(name)


def blocks(count, threads):
    return int(min((count + threads - 1) // threads, 65535))


_KEEP = []


def run(mod, name, grid, block, args, shared=0):
    H.launch(fn(mod, name), grid, block, shared, args)
    H.sync()
    _KEEP.clear()


def ptr(b):
    """A buffer's address as a kernel argument; the buffer is held until the launch has finished (an inline
    Buf.of(...) would otherwise be freed before the kernel runs)."""
    _KEEP.append(b)
    return u64(b.ptr)


def t_casts():
    allh = np.arange(65536, dtype=np.uint16)
    out = Buf(65536 * 4)
    run("dsv41_ops", "tf_ds_f16_to_f32_kernel", (blocks(65536, 256), 1, 1), (256, 1, 1),
        [ptr(Buf.of(allh)), ptr(out), u64(65536)])
    got = out.download(65536, np.float32)
    want = allh.view(np.float16).astype(np.float32)
    C.record(kernel="tf_ds_f16_to_f32_kernel", case="all 65536 fp16", ok=bool(np.array_equal(got.view(np.uint32),
             want.view(np.uint32)) or np.array_equal(np.isnan(got), np.isnan(want)) and np.array_equal(
             got[~np.isnan(got)], want[~np.isnan(want)])))
    run("pointwise", "tf_bf16_to_f32_kernel", (blocks(65536, 256), 1, 1), (256, 1, 1),
        [ptr(Buf.of(allh)), ptr(out), u64(65536)])
    got = out.download(65536, np.uint32)
    C.record(kernel="tf_bf16_to_f32_kernel", case="all 65536 bf16", ok=bool(np.array_equal(got, allh.astype(np.uint32) << 16)))
    # f32 -> bf16: random bits over the whole range, every tie pattern, specials
    n = 1 << 22
    bits = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    bits[: 1 << 16] = (rng.integers(0, 1 << 16, 1 << 16).astype(np.uint32) << 16) | 0x8000     # exact ties
    x = bits.view(np.float32)
    ob = Buf(n * 2)
    run("pointwise", "tf_f32_to_bf16_kernel", (blocks(n, 256), 1, 1), (256, 1, 1), [ptr(Buf.of(x)), ptr(ob), u64(n)])
    got = ob.download(n, np.uint16)
    fin = np.isfinite(x)
    want = C.bf16_bits(R.bf16(x[fin].astype(np.float64)))
    ok_f = np.array_equal(got[fin], want)
    nan_ok = bool(np.all(((got[np.isnan(x)] & 0x7F80) == 0x7F80) & ((got[np.isnan(x)] & 0x7F) != 0)))
    inf_ok = bool(np.array_equal(got[np.isinf(x)], C.bf16_bits(x[np.isinf(x)].astype(np.float64))))
    C.record(kernel="tf_f32_to_bf16_kernel", case=f"{n} fp32 (65536 exact ties), RNE", ok=bool(ok_f and nan_ok and inf_ok),
             finite_mismatch=int((got[fin] != want).sum()), nan_ok=nan_ok, inf_ok=inf_ok)
    # the add of two ranks' partials then bf16
    g0, g1 = (rng.standard_normal(n) * 10).astype(np.float32), (rng.standard_normal(n) * 10).astype(np.float32)
    run("dsv41_ops", "tf_ds_add2_bf16_kernel", (blocks(n, 256), 1, 1), (256, 1, 1),
        [ptr(Buf.of(g0)), ptr(Buf.of(g1)), ptr(ob), u64(n)])
    want = C.bf16_bits(R.bf16(R.f32(g0.astype(np.float64) + g1.astype(np.float64))))
    C.record(kernel="tf_ds_add2_bf16_kernel", case=f"{n}", ok=bool(np.array_equal(ob.download(n, np.uint16), want)))
    # wl.to(bf16) * s -> bf16
    s = np.float32(128 ** -0.5 * 32 ** -0.5)
    run("dsv41_torch", "tf_ds_bf16_scale_kernel", (blocks(n, 256), 1, 1), (256, 1, 1),
        [ptr(Buf.of(g0)), ptr(ob), u64(n), f32(s)])
    want = C.bf16_bits(R.bf16(R.f32(R.bf16(g0.astype(np.float64)) * np.float64(s))))
    C.record(kernel="tf_ds_bf16_scale_kernel", case=f"{n}", ok=bool(np.array_equal(ob.download(n, np.uint16), want)))


def t_moves():
    rows, rb, ld = 300, 1000, 1024
    src = rng.integers(0, 256, (rows, ld), dtype=np.uint8)
    dst0 = rng.integers(0, 256, (rows * 2, ld + 16), dtype=np.uint8)
    idx = rng.permutation(rows * 2)[:rows].astype(np.int64)
    d = Buf.of(dst0)
    run("dsv41_ops", "tf_ds_scatter_rows_kernel", (rows, 1, 1), (256, 1, 1),
        [ptr(Buf.of(src)), i64(ld), ptr(Buf.of(idx)), ptr(d), i64(ld + 16), i64(rb), i32(rows)])
    want = dst0.copy()
    want[idx, :rb] = src[:, :rb]
    C.record(kernel="tf_ds_scatter_rows_kernel", case="unaligned rows", ok=bool(np.array_equal(d.download(dst0.shape, np.uint8), want)))
    # 16-byte rows
    src16 = rng.integers(0, 256, (rows, 1024), dtype=np.uint8)
    dst16 = np.zeros((rows, 2048), np.uint8)
    d16 = Buf.of(dst16)
    run("dsv41_ops", "tf_ds_copy_rows16_kernel", (blocks(1024 // 16, 256), rows, 1), (256, 1, 1),
        [ptr(Buf.of(src16)), u64(1024 // 16), ptr(d16), u64(2048 // 16), u64(1024 // 16)])
    want = dst16.copy()
    want[:, :1024] = src16
    C.record(kernel="tf_ds_copy_rows16_kernel", case="rows of 1024 B", ok=bool(np.array_equal(d16.download(dst16.shape, np.uint8), want)))
    # strided copy (a row a block)
    sb = Buf.of(src)
    db = Buf.of(np.zeros((rows, ld), np.uint8))
    run("movement", "tf_strided_copy_kernel", (rows, 1, 1), (256, 1, 1),
        [ptr(sb), ptr(db), u64(rows), u64(1), u64(rb), u64(ld), u64(rb), u64(ld), u64(rb)])
    want = np.zeros((rows, ld), np.uint8)
    want[:, :rb] = src[:, :rb]
    C.record(kernel="tf_strided_copy_kernel", case="rows", ok=bool(np.array_equal(db.download((rows, ld), np.uint8), want)))
    # gather rows with an invalid index flagged
    gi = rng.integers(0, rows, 50).astype(np.int64)
    gi[7] = rows + 3
    gd0 = rng.integers(0, 256, (50, ld), dtype=np.uint8)
    gd, inv = Buf.of(gd0), Buf.of(np.zeros(1, np.uint32))
    run("movement", "tf_gather_rows_kernel", (50, 1, 1), (256, 1, 1),
        [ptr(sb), ptr(gd), ptr(Buf.of(gi)), u64(50), u64(rows), u64(ld), ptr(inv)])
    want = src[np.clip(gi, 0, rows - 1)].copy()
    want[7] = gd0[7]
    C.record(kernel="tf_gather_rows_kernel", case="50 rows, one invalid", ok=bool(
        np.array_equal(gd.download((50, ld), np.uint8), want) and inv.download(1, np.uint32)[0] == 1))


def keys_of(score, cols):
    s = np.asarray(score, np.float32) + np.float32(0)
    b = s.view(np.int32).astype(np.int64)
    return (np.where(b < 0, b ^ 0x7FFFFFFF, b) << 32) | (0xFFFFFFFF - np.asarray(cols, np.int64))


def t_select():
    for rows, n, k in ((3, 131072, 512), (8, 32768, 512), (2, 600, 512), (4, 5000, 2048)):
        s = rng.standard_normal((rows, n)).astype(np.float32)
        s[rng.random((rows, n)) < 0.3] = -np.inf
        s[:, :4] = 0.25
        keys = keys_of(s, np.arange(n)[None, :])
        top = Buf(rows * k * 8)
        run("dsv41_ops", "tf_ds_topk_i64_kernel", (rows, 1, 1), (1024, 1, 1),
            [ptr(Buf.of(keys)), i64(n), i32(n), i32(k), ptr(top)])
        got = np.sort(top.download((rows, k), np.int64), 1)
        want = np.sort(keys, 1)[:, -k:]
        C.record(kernel="tf_ds_topk_i64_kernel", case=f"rows={rows} n={n} k={k}", ok=bool(np.array_equal(got, want)))
        if k <= 2048:
            vis = rng.integers(1, n + 1, rows).astype(np.int64)
            out = Buf(rows * k * 8)
            run("dsv41_ops", "tf_ds_topk_indices_kernel", (rows, 1, 1), (1024, 1, 1),
                [ptr(Buf.of(s)), i64(n), i32(n), i32(k), ptr(Buf.of(vis)), ptr(out)])
            idx = np.sort(0xFFFFFFFF - (want & 0xFFFFFFFF), 1)
            wantv = np.where(idx < vis[:, None], idx, -1)
            C.record(kernel="tf_ds_topk_indices_kernel", case=f"rows={rows} n={n} k={k}",
                     ok=bool(np.array_equal(out.download((rows, k), np.int64), wantv)))
    # topk_i64 with equal keys (the pruned path's tile maxima: pad tiles all INT64_MIN) and a repeated k-th key, rows
    # held in registers (n <= 32768) and rows read again each digit (n > 32768); and 2048-row prompt shapes
    for rows, n, k, pads in ((4, 1024, 512, 700), (3, 600, 512, 300), (2, 40000, 512, 39700), (3, 8192, 512, 0),
                             (2048, 8192, 512, 0), (64, 32768, 512, 100), (16, 1000, 1000, 0)):
        keys = rng.integers(-2**62, 2**62, (rows, n)).astype(np.int64)
        if pads:
            keys[:, rng.permutation(n)[:pads]] = np.iinfo(np.int64).min
        if n == 600:
            keys[:, 100:140] = keys[:, 99:100]                           # forty copies of one key
        top = Buf(rows * k * 8)
        run("dsv41_ops", "tf_ds_topk_i64_kernel", (rows, 1, 1), (1024, 1, 1),
            [ptr(Buf.of(keys)), i64(n), i32(n), i32(k), ptr(top)])
        got = np.sort(top.download((rows, k), np.int64), 1)
        want = np.sort(keys, 1)[:, -k:]
        C.record(kernel="tf_ds_topk_i64_kernel", case=f"equal keys / prompt rows={rows} n={n} k={k} pads={pads}",
                 ok=bool(np.array_equal(got, want)))
    # the candidate pool helpers
    rows, width, bsize = 5, 20000, 8
    s = rng.standard_normal((rows, width)).astype(np.float32)
    s[rng.random((rows, width)) < 0.5] = -np.inf
    s[0, 16:24] = -np.inf
    s[1, 40] = np.nan
    vis = np.array([1, 900, width, 17, 8], np.int64)
    nb = (width + bsize - 1) // bsize
    out = Buf(rows * nb * 4)
    run("dsv41_ops", "tf_ds_block_max_kernel", (blocks(nb, 256), rows, 1), (256, 1, 1),
        [ptr(Buf.of(s)), i64(width), i32(width), i32(bsize), ptr(Buf.of(vis)), ptr(out), i64(nb), i32(nb)])
    pad = np.full((rows, nb * bsize), -np.inf, np.float32)
    pad[:, :width] = s
    blk = pad.reshape(rows, nb, bsize)
    want = np.where(np.isnan(blk).any(-1), np.nan, np.max(np.where(np.isnan(blk), -np.inf, blk), -1)).astype(np.float32)
    last = (vis - 1) // bsize
    want[np.arange(rows), last] = np.inf
    got = out.download((rows, nb), np.float32)
    C.record(kernel="tf_ds_block_max_kernel", case=f"rows={rows} width={width}",
             ok=bool(np.array_equal(np.isnan(got), np.isnan(want)) and np.array_equal(got[~np.isnan(got)], want[~np.isnan(want)])))
    bm = np.nan_to_num(want, nan=-np.inf)
    k = 64
    tops = np.stack([np.sort(rng.permutation(nb)[:k]) for _ in range(rows)]).astype(np.int64)
    mask0 = np.full((rows, nb + 16), 7, np.uint8)
    mk, cb = Buf.of(mask0), Buf(rows * k * 4)
    run("dsv41_ops", "tf_ds_pool_pick_kernel", (rows, 1, 1), (1024, 1, 1),
        [ptr(Buf.of(bm.astype(np.float32))), i64(nb), i32(nb), ptr(Buf.of(tops)), i32(k), ptr(mk), i64(nb + 16), ptr(cb)])
    okp = np.take_along_axis(bm, tops, 1) > -np.inf
    wm = mask0.copy()
    wm[:, :nb] = 0
    for r in range(rows):
        wm[r, tops[r][okp[r]]] = 1
    wc = np.where(okp, tops, -1).astype(np.int32)
    C.record(kernel="tf_ds_pool_pick_kernel", case=f"rows={rows} k={k}", ok=bool(
        np.array_equal(mk.download(mask0.shape, np.uint8), wm) and np.array_equal(cb.download((rows, k), np.int32), wc)))
    sc = Buf.of(s)
    mask = (rng.random((rows, nb + 16)) < 0.5).astype(np.uint8)
    run("dsv41_ops", "tf_ds_apply_pool_kernel", (blocks(width, 256), rows, 1), (256, 1, 1),
        [ptr(sc), i64(width), i32(width), i32(bsize), ptr(Buf.of(mask)), i64(nb + 16)])
    keep = np.repeat(mask[:, :nb], bsize, 1)[:, :width].astype(bool)
    want = np.where(keep, s, -np.inf)
    got = sc.download((rows, width), np.float32)
    C.record(kernel="tf_ds_apply_pool_kernel", case=f"rows={rows}", ok=bool(np.array_equal(np.isnan(got), np.isnan(want)) and
             np.array_equal(got[~np.isnan(got)], want[~np.isnan(want)])))
    ob = Buf(rows * (nb + 16))
    run("dsv41_ops", "tf_ds_cand_fast_kernel", (blocks(nb, 256), rows, 1), (256, 1, 1),
        [ptr(Buf.of(pad)), i64(nb * bsize), i32(nb), i32(bsize), ptr(Buf.of(vis)), ptr(ob), i64(nb + 16)])
    wantc = ((~np.isnan(blk).any(-1)) & (np.where(np.isnan(blk), -np.inf, blk) > -np.inf).any(-1))
    wantc[np.arange(rows), last] = True
    C.record(kernel="tf_ds_cand_fast_kernel", case=f"rows={rows}", ok=bool(
        np.array_equal(ob.download((rows, nb + 16), np.uint8)[:, :nb], wantc.astype(np.uint8))))
    io = Buf(rows * 3000 * 8)
    run("dsv41_ops", "tf_ds_iota_vis_kernel", (blocks(3000, 256), rows, 1), (256, 1, 1), [ptr(Buf.of(vis)), ptr(io), i32(3000)])
    j = np.arange(3000)[None, :]
    C.record(kernel="tf_ds_iota_vis_kernel", case=f"rows={rows}", ok=bool(
        np.array_equal(io.download((rows, 3000), np.int64), np.where(j < vis[:, None], j, -1))))


def t_argmax():
    rows, n = 9, 129280
    lg = rng.standard_normal((rows, n)).astype(np.float32)
    lg[1, 5000] = lg[1, 7000] = 50.0              # a tie: the first index
    lg[2, 0] = np.nan                             # a NaN first: 0
    lg[3, 77] = np.nan                            # a NaN elsewhere: never larger
    lg[4] = -np.inf
    out = Buf(rows * 4)
    run("dsv41_ops", "tf_ds_argmax_rows_kernel", (rows, 1, 1), (1024, 1, 1), [ptr(Buf.of(lg)), u64(n), i32(n), ptr(out)])
    want = [0 if np.isnan(lg[r, 0]) else int(np.argmax(np.where(np.isnan(lg[r]), -np.inf, lg[r]))) for r in range(rows)]
    C.record(kernel="tf_ds_argmax_rows_kernel", case=f"rows={rows} vocab={n} (ties, NaN, -inf rows)",
             ok=bool(np.array_equal(out.download(rows, np.uint32), np.array(want, np.uint32))), got=out.download(rows, np.uint32).tolist(), want=want)


def t_torch_math():
    # ops.rms_norm with torch's reduction layout: exact.zig's rmsLanes / grid
    for rows, d in ((1, 512), (16, 512), (1024, 512), (3, 128), (2048, 128)):
        dim0 = d // 4
        d0 = 1 << (dim0.bit_length() - 1) if dim0 < 512 else 512
        d1 = 1 << (rows.bit_length() - 1) if rows < 512 else 512
        bw = min(d0, 32)
        bh = min(d1, 512 // bw)
        lanes = min(d0, 512 // bh)
        h = max(1, min(512 // lanes, 64))
        x = C.rand_bf16(rng, (rows, d), 2.0)
        w = C.rand_bf16(rng, (d,), 0.5)
        out = Buf(rows * d * 2)
        factor = np.float32(rows) / np.float32(rows * d)
        run("dsv41_torch", "tf_ds_rms_norm_kernel", ((rows + h - 1) // h, 1, 1), (lanes, h, 1),
            [ptr(Buf.of(x)), i64(d), ptr(Buf.of(w)), ptr(out), i64(d), i32(rows), i32(d), f32(factor), f32(1e-20)],
            shared=lanes * h * 4 if lanes > 32 else 0)
        xs, ws = C.from_bf16_bits(x), C.from_bf16_bits(w)
        r_ = 1 / np.sqrt(np.mean(xs * xs, -1, keepdims=True) + 1e-20)
        exact = ws * (xs * r_)
        bound = C.gamma(d + 8) * np.abs(exact)
        m = C.cmp_rounded(C.from_bf16_bits(out.download((rows, d), np.uint16)), exact, R.bf16(exact), C.ulp_bf16, bound=bound)
        C.record(kernel="tf_ds_rms_norm_kernel", case=f"rows={rows} d={d} block=({lanes},{h})", **m)
    # _compress at ratio 2
    G, Cc = 700, 512
    kv = rng.standard_normal((G, 2, Cc)).astype(np.float32)
    sc = (rng.standard_normal((G, 2, Cc)) * 3).astype(np.float32)
    out = Buf(G * Cc * 2)
    run("dsv41_torch", "tf_ds_compress2_kernel", (min(4096, blocks(G * Cc, 256)), 1, 1), (256, 1, 1),
        [ptr(Buf.of(kv)), ptr(Buf.of(sc)), ptr(out), i32(G), i32(Cc)])
    s64 = sc.astype(np.float64)
    p = np.exp(s64 - s64.max(1, keepdims=True))
    p = p / p.sum(1, keepdims=True)
    terms = kv.astype(np.float64) * p
    exact = terms.sum(1)
    bound = C.gamma(12) * np.abs(terms).sum(1)
    m = C.cmp_rounded(C.from_bf16_bits(out.download((G, Cc), np.uint16)), exact, R.bf16(exact), C.ulp_bf16, bound=bound)
    C.record(kernel="tf_ds_compress2_kernel", case=f"G={G} C={Cc}", **m)
    # a DSpark tap's mean over the 4 streams
    rows, d = 33, 5120
    hh = C.rand_bf16(rng, (rows, 4, d))
    out = Buf(rows * 6000 * 2)
    run("dsv41_torch", "tf_ds_hc_mean4_kernel", (min(4096, blocks(rows * d, 256)), 1, 1), (256, 1, 1),
        [ptr(Buf.of(hh)), ptr(out), i64(6000), i32(rows), i32(d)])
    exact = C.from_bf16_bits(hh).mean(1)
    got = C.from_bf16_bits(out.download((rows, 6000), np.uint16))[:, :d]
    m = C.cmp_rounded(got, exact, R.bf16(exact), C.ulp_bf16, bound=C.gamma(4) * np.abs(C.from_bf16_bits(hh)).sum(1) / 4)
    C.record(kernel="tf_ds_hc_mean4_kernel", case=f"rows={rows}", **m)
    # ops.rope_ (in the image; the text path does not call it): complex multiply, bf16 parts
    n, half, off, ldx = 40, 32, 448, 512
    x = C.rand_bf16(rng, (n, ldx))
    cos = rng.standard_normal((100, half)).astype(np.float32)
    sin = rng.standard_normal((100, half)).astype(np.float32)
    pos = rng.integers(0, 100, n).astype(np.int64)
    xb = Buf.of(x)
    run("dsv41_torch", "tf_ds_rope_kernel", (min(4096, blocks(n * half, 256)), 1, 1), (256, 1, 1),
        [ptr(xb), i64(ldx), i32(off), ptr(Buf.of(cos)), ptr(Buf.of(sin)), ptr(Buf.of(pos)), i32(n), i32(half), i32(0)])
    xs = C.from_bf16_bits(x)
    seg = xs[:, off:off + 2 * half]
    exact = R.rope_pairs(seg, cos[pos].astype(np.float64), sin[pos].astype(np.float64))
    got = C.from_bf16_bits(xb.download((n, ldx), np.uint16))[:, off:off + 2 * half]
    bound = C.gamma(2) * (np.abs(np.repeat(seg[:, 0::2] * cos[pos], 2, 1)) + np.abs(np.repeat(seg[:, 1::2] * sin[pos], 2, 1)))
    m = C.cmp_rounded(got, exact, R.bf16(exact), C.ulp_bf16, bound=bound * 2)
    C.record(kernel="tf_ds_rope_kernel", case=f"n={n}", **m)


def t_clock_prefetch():
    buf = Buf.of(np.zeros(1 + 2 * 8, np.uint64))
    for i in range(4):
        run("dsv41_ops", "tf_ds_clock_kernel", (1, 1, 1), (32, 1, 1), [ptr(buf), u32(i + 1), u32(1 if i == 0 else 0), u32(8)])
    b = buf.download(17, np.uint64)
    ts = b[1:1 + 2 * 4:2].astype(np.int64)
    C.record(kernel="tf_ds_clock_kernel", case="4 stamps", ok=bool(b[0] == 4 and np.all(np.diff(ts) > 0) and
             np.array_equal(b[2:2 + 2 * 4:2], [1, 2, 3, 4])), deltas_ns=np.diff(ts).tolist())

    class Pf(ctypes.Structure):
        _fields_ = [("ptr", ctypes.c_uint64 * 16), ("bytes", ctypes.c_uint64 * 16), ("n", ctypes.c_int)]
    pf = Pf()
    pf.n = 1
    pf.ptr[0] = buf.ptr
    pf.bytes[0] = 128
    run("dsv41_ops", "tf_ds_l2_prefetch_kernel", (1, 1, 1), (128, 1, 1), [pf, i64(3000), i64(4558)])
    C.record(kernel="tf_ds_l2_prefetch_kernel", case="no-op launch (struct argument by value)", ok=True)


def t_hipblas():
    """torch.matmul's fp32 GEMMs (cublas.zig xwT / xv) on hipBLAS with the same column-major call shapes."""
    lib = ctypes.CDLL("/opt/rocm/lib/libhipblas.so")
    h = ctypes.c_void_p()
    assert lib.hipblasCreate(ctypes.byref(h)) == 0
    T, N = 112, 111
    one, zero = f32(1.0), f32(0.0)
    for rows, k, n in ((512, 5120, 384), (2048, 5120, 384), (2048, 5120, 32), (8192, 5120, 384), (5, 5376, 1)):
        x = rng.standard_normal((rows, k)).astype(np.float32)
        w = (rng.standard_normal((n, k)) * 0.02).astype(np.float32)
        out = Buf(rows * n * 4)
        xb, wb = Buf.of(x), Buf.of(w)
        call = lambda: lib.hipblasSgemm(h, T, N, n, rows, k, ctypes.byref(one), ctypes.c_void_p(wb.ptr), k,
                                         ctypes.c_void_p(xb.ptr), k, ctypes.byref(zero), ctypes.c_void_p(out.ptr), n)
        if n == 1:
            call = lambda: lib.hipblasSgemm(h, N, N, 1, rows, k, ctypes.byref(one), ctypes.c_void_p(wb.ptr), 1,
                                             ctypes.c_void_p(xb.ptr), k, ctypes.byref(zero), ctypes.c_void_p(out.ptr), 1)
        st = call()
        H.sync()
        got = out.download((rows, n), np.float32).astype(np.float64)
        st2 = call()
        H.sync()
        again = out.download((rows, n), np.float32).astype(np.float64)
        ref = x.astype(np.float64) @ w.astype(np.float64).T
        bnd = C.gamma(k) * (np.abs(x.astype(np.float64)) @ np.abs(w.astype(np.float64)).T)
        m = C.cmp_bound(got, ref, bnd)
        m["deterministic"] = bool(np.array_equal(got, again))
        m["status"] = [st, st2]
        m["ms"] = round(H.time_ms(call, iters=20), 4)
        m["ok"] = m["ok"] and m["deterministic"] and st == 0
        C.record(kernel="hipblasSgemm (cublas xwT/xv)", case=f"rows={rows} k={k} n={n}", **m)
    lib.hipblasDestroy(h)


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for t in (t_casts, t_moves, t_select, t_argmax, t_torch_math, t_clock_prefetch, t_hipblas):
        if only and t.__name__ not in only:
            continue
        t()
    H.sync()
    sys.exit(C.done())
