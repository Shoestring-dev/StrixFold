"""Two aux_index.hip builds give the same bits (fp32 scores and int64 keys + tile maxima) on random codes with every
E8M0 scale byte 0..255 (the table decode's fast path covers 2..252, the rest takes the float path)."""
import ctypes, os, sys
import numpy as np
import common as C, hiprt as H, t_hipindex as T
from hiprt import Buf
C.log_open("x_index_same")
A = H.Module(open(sys.argv[1], "rb").read()).function("tf_ds_aux_index_rows_h32_d128")
B = H.Module(open(sys.argv[2], "rb").read()).function("tf_ds_aux_index_rows_h32_d128")
rng = np.random.default_rng(31)
C.gpu_lock()
ok = True
for rows, n, flags in ((40, 4096, 0), (40, 4096, 3), (9, 1000, 3)):
    codes = rng.integers(0, 256, (n, 64), dtype=np.uint8)
    scales = (np.arange(n * 4) % 256).astype(np.uint8).reshape(n, 4)
    rng.shuffle(scales)
    q, w = C.rand_bf16(rng, (rows, 32, 128), 0.5), C.rand_bf16(rng, (rows, 32), 0.2)
    vis = rng.integers(1, n + 1, rows).astype(np.int64)
    nt = (n + 63) // 64
    res = []
    for f in (A, B):
        b = [Buf.of(q), Buf.of(codes), Buf.of(scales), Buf.of(w), Buf.of(vis)]
        out, tm = Buf(rows * n * 8), Buf(rows * nt * 8)
        out.fill(0x5A)
        tm.fill(0x5A)
        a = T.IndexArgs(q=b[0].ptr, k=b[1].ptr, ks=b[2].ptr, w=b[3].ptr, vis=b[4].ptr, out=out.ptr, tmax=tm.ptr, rows=rows,
                        n=n, nt=nt, flags=flags)
        H.launch(f, ((rows + 7) // 8, T.splits_for(rows, nt), 1), (256, 1, 1), 0, [a])
        H.sync()
        res.append((out.download((rows, n * 8), np.uint8), tm.download((rows, nt * 8), np.uint8)))
    same = bool(np.array_equal(res[0][0], res[1][0]) and (flags != 3 or np.array_equal(res[0][1], res[1][1])))
    ok &= same
    C.record(kernel="aux.index_rows", case=f"builds agree rows={rows} n={n} flags={flags}", ok=same)
sys.exit(C.done())
