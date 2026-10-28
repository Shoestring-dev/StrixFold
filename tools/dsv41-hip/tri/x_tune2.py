"""Tuning experiment 2: the prompt indexer (_index_score_rows) by rows-a-program (RB) and warps: time, TFLOPS,
registers; tile maxima equal to the stored keys' maxima; scores bit-equal to the production setting (RB 1, 2 warps)."""
import copy, json
import numpy as np
import common as C, hiprt as H
from hiprt import Buf, P
import x_tune as X        # (runs its experiment on import: guarded below by a flag)

L = json.load(open(C.ROOT / "upstream" / "launches.json"))
rng = np.random.default_rng(9)
C.log_open("x_tune2")
isr = L["_index_score_rows"][5]
rows, n = 2048, 16384
nt = n // 64
raw = rng.standard_normal((n, 128)) * np.exp(rng.uniform(-2, 2, (n, 1)))
import refs as R
deq, codes, sc = R.fp4_quant_rows(raw, 32, False)
codes = R.fp4_pack(codes)
q, w = Buf.of(C.rand_bf16(rng, (rows, 32, 128), 0.5)), Buf.of(C.rand_bf16(rng, (rows, 32), 0.2))
vis = Buf.of(np.sort(rng.integers(n // 2, n + 1, rows)).astype(np.int64))
kb, sb = Buf.of(codes), Buf.of(sc)
base_keys = None
for rb, warps in ((1, 2), (1, 4), (2, 4), (4, 4), (4, 8), (8, 8), (2, 8)):
    c = copy.deepcopy(isr); c["num_warps"] = warps
    kv = X.variant("_index_score_rows", c, warps)
    kv2 = C.extra_variant("_index_score_rows", c, RB=rb) if rb != 1 else kv
    if rb != 1:
        import aot_build
        old = dict(aot_build.EXTRA_OPTIONS.get("_index_score_rows", {}))
        aot_build.EXTRA_OPTIONS["_index_score_rows"] = {}
        C._EXTRA.clear()
        kv2 = C.extra_variant("_index_score_rows", c, RB=rb)
        aot_build.EXTRA_OPTIONS["_index_score_rows"] = old
    out, tm = Buf(rows * n * 8), Buf(rows * nt * 8)
    a = dict(Q=P(q, "*bf16"), K=P(kb, "*u8"), KS=P(sb, "*u8"), Wt=P(w, "*bf16"), VIS=P(vis, "*i64"), OUT=P(out, "*i64"),
             n=n, rows=rows, TMAX=P(tm, "*i64"), nt=nt)
    call = lambda: C.run_variant(kv2, ((rows + rb - 1) // rb, nt, 1), a)
    call(); H.sync()
    keys = out.download((rows, n), np.int64)
    tmax = tm.download((rows, nt), np.int64)
    tm_ok = bool(np.array_equal(tmax, keys.reshape(rows, nt, 64).max(-1)))
    if base_keys is None:
        base_keys = keys
    same = bool(np.array_equal(keys, base_keys))
    ms = H.time_ms(call, iters=3, warmup=1)
    C.record(kernel="_index_score_rows", case=f"RB={rb} warps={warps}", us=round(ms * 1000), TFLOPS=round(2 * rows * n * 32 * 128 / (ms * 1e-3) / 1e12, 2),
             tmax_consistent=tm_ok, keys_equal_production=same, vgpr=kv2["isa_stats"].get("vgpr_count"), spill=kv2["isa_stats"].get("vgpr_spill_count"))
C.done()
