"""WMMA throughput / latency on gfx1151 (wmma_rate.hip): TFLOPS with 8 independent accumulators a wave, and the
dependent-chain rate, for 1..8 waves a SIMD (HIP events)."""
import ctypes, os
import common as C, hiprt as H
from hiprt import Buf
C.log_open("x_wmma_rate")
M = H.Module(open(os.environ["WR_HSACO"], "rb").read())
out = Buf(64)
C.gpu_lock()
C.quiet_window("w-tri-wmma")
iters = 2000
for name in ("wmma_rate", "wmma_chain"):
    f = M.function(name)
    for wgs in (20, 40, 80, 160, 320, 640):
        ms = H.time_ms(lambda: H.launch(f, (wgs, 1, 1), (256, 1, 1), 0, [ctypes.c_uint64(out.ptr), ctypes.c_int32(iters)]), iters=3, warmup=1)
        n = wgs * 8 * iters * 8                                   # WMMAs
        print(f"{name:10s} wgs={wgs:4d} waves/SIMD~{wgs * 8 / 80:5.1f} {ms:8.3f} ms  {n * 8192 / ms / 1e9:7.2f} TFLOPS  {n / 80 / (ms * 1e-3) / 1e9:6.3f} G wmma/s/SIMD", flush=True)
C.done()
