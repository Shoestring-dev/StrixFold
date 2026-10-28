"""Streaming bandwidth probe (bw_probe.hip): 82 MB writes (8 / 16 B per lane), reads, copy, and the mHC shape
(82 MB + 164 MB reads, 82 MB write). Run under rocprofv3; x_runs.py prints the runs in order."""
import ctypes, os
import common as C, hiprt as H
from hiprt import Buf
C.log_open("x_bw")
M = H.Module(open(os.environ["BW_HSACO"], "rb").read())
NB = 2048 * 4 * 5120 * 2                                    # xo of a 2048-row chunk
C.gpu_lock()
a, b, o, sink = Buf(NB), Buf(2 * NB), Buf(NB), Buf(16)
C.quiet_window("w-tri-bw")
for grid in (1280, 4096):
    for name, args in (("bw_write16", [ctypes.c_uint64(o.ptr), ctypes.c_int64(NB // 16)]),
                       ("bw_write8", [ctypes.c_uint64(o.ptr), ctypes.c_int64(NB // 8)]),
                       ("bw_read16", [ctypes.c_uint64(a.ptr), ctypes.c_uint64(sink.ptr), ctypes.c_int64(NB // 16)]),
                       ("bw_copy16", [ctypes.c_uint64(a.ptr), ctypes.c_uint64(o.ptr), ctypes.c_int64(NB // 16)]),
                       ("bw_rrw16", [ctypes.c_uint64(a.ptr), ctypes.c_uint64(b.ptr), ctypes.c_uint64(o.ptr), ctypes.c_int64(NB // 16)])):
        f = M.function(name)
        for _ in range(5):
            H.launch(f, (grid, 1, 1), (256, 1, 1), 0, args)
        H.sync()
C.done()
