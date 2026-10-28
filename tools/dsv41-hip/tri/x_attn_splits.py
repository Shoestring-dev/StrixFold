"""Decode / verify: aux.sparse_attn device time against the split count (ring window + 512 FP4 picks), and the Triton
merge of 8 splits for scale. HIP events, rounds interleaved."""
import statistics
import common as C, hiprt as H, t_hipattn as T, x_hipattn_perf as XP
C.log_open("x_attn_splits")
C.gpu_lock()
cases = []
for rows in (1, 2, 4, 6, 16):
    d = XP.inputs(rows, False)
    per = {}
    for splits in (4, 8, 16, 20, 40):
        keep, cap = [], {}
        real = H.launch
        def grab(f, grid, block, shared, args):
            cap["a"], cap["grid"] = args[0], grid
        H.launch = grab
        T.hip_attn(d, splits, keep)
        H.launch = real
        per[splits] = (cap["a"], cap["grid"], keep)
    cases.append((rows, per))
print("load", C.quiet_window("w-tri-splits"), flush=True)
for rows, per in cases:
    t = {s: [] for s in per}
    for _ in range(9):
        for s, (a, grid, _) in per.items():
            t[s].append(H.time_ms(lambda: H.launch(T.FN, grid, (T.BLOCK, 1, 1), 0, [a]), iters=20, warmup=2))
    med = {s: statistics.median(v) * 1e3 for s, v in t.items()}
    print(f"rows={rows:3d} " + "  ".join(f"splits {s:2d}: {v:6.1f} us" for s, v in med.items()), flush=True)
    C.record(kernel="aux.sparse_attn", case=f"rows={rows} splits", ok=None, **{f"s{s}": v for s, v in med.items()})
C.done()
