"""Device time of the HIP mHC prompt mixes (aux.hc_mix_pf, posted, world 2) against the Triton _hc_mix_pf (RB 8) for
512-8192-row chunks. Launches only; run under rocprofv3 --kernel-trace (this process takes the GPU lock and a quiet
window itself) and read the runs with x_runs.py. Bytes moved per row: x + xo (bf16) + the two ranks' fp32 partials."""
import json
import numpy as np
import common as C, hiprt as H, t_hc as TH, t_hiphc as T
from hiprt import Buf, P
C.log_open("x_hiphc_perf")
L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
kpf = TH.consts_of(L["_hc_mix_pf"][1])
rng = np.random.default_rng(8)
D = 5120
C.gpu_lock()
cases = []
for rows in (512, 2048, 8192):
    x = C.rand_bf16(rng, (rows, 4, D), 1.0)
    g = (rng.standard_normal((2, rows, D)) * 0.5).astype(np.float32)
    post = (2 * rng.random((rows, 4))).astype(np.float32)
    comb = rng.random((rows, 4, 4)).astype(np.float32)
    cases.append((rows, x, g, post, comb))
C.record(kernel="window", case="quiet", ok=None, load=C.quiet_window("w-tri-hc"))
for rows, x, g, post, comb in cases:
    xb, gb, pb, cb = Buf.of(x), Buf.of(g), Buf.of(post), Buf.of(comb)
    xo, part = Buf(rows * 4 * D * 2), Buf(rows * 40 * 32 * 4)
    a = T.HcArgs(x=xb.ptr, xo=xo.ptr, g=gb.ptr, post=pb.ptr, comb=cb.ptr, fn=TH.FN_B.ptr, part=part.ptr, rs=rows * D,
                 rows=rows, world=2, flags=1)
    C.record(kernel="case", case=f"rows={rows} hip", ok=None, mbytes=rows * (4 * D * 2 * 2 + 2 * D * 4) / 1e6)
    for _ in range(5):
        H.launch(T.FN, (((rows + T.RB - 1) // T.RB) * 10, 1, 1), (256, 1, 1), 0, [a])
    H.sync()
    C.record(kernel="case", case=f"rows={rows} triton", ok=None)
    args = dict(X=P(xb, "*bf16"), XO=P(xo, "*bf16"), G=P(gb, "*fp32"), RS=rows * D, POST=P(pb, "*fp32"),
                COMB=P(cb, "*fp32"), FN=P(TH.FN_B, "*fp32"), PART=P(part, "*fp32"), rows=rows)
    for _ in range(5):
        C.KIT.run("_hc_mix_pf", (((rows + 7) // 8) * 40, 1, 1), args, kpf)
    H.sync()
C.done()
