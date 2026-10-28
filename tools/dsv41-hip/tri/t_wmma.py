"""gfx1151's v_wmma_f32_16x16x16_bf16 (wave32) as aux_attn.hip uses it: the operand layouts (A: lane l holds row l % 16,
B: lane l holds column l % 16, D: lane l holds column l % 16, rows 2 i + l / 16) and the accumulation's rounding.
Builds wmma_probe.hip with the kit flags. Finding (2026-10-19): products of exact bf16 integers summed with zeros come
back 1 ulp low for negative results (I @ B = B - 1 ulp where B < 0): the fp32 accumulation is not IEEE round-to-
nearest; a WMMA kernel's error stays within the gamma bound, but exact-arithmetic tests cannot go through WMMA."""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import common as C
import hiprt as H
from hiprt import Buf

C.log_open("t_wmma")
SRC = Path(__file__).resolve().parent / "wmma_probe.hip"
OBJ = C.ROOT / "build" / "wmma_probe.hsaco"
OBJ.parent.mkdir(exist_ok=True)
subprocess.run(["/usr/bin/hipcc", "--offload-arch=gfx1151", "--offload-device-only", "--no-gpu-bundle-output", "-O3",
                "-std=c++20", "-fno-gpu-rdc", "-mno-wavefrontsize64", "-ffp-contract=off", "-fno-fast-math",
                "-fdenormal-fp-math=ieee", str(SRC), "-o", str(OBJ)], check=True,
               env={k: v for k, v in os.environ.items() if not k.startswith("HIPCC_")})
f = H.Module(OBJ.read_bytes()).function("tf_ds_aux_wmma_probe")


def run(a, b):
    ab, bb, db = Buf.of(a.astype(np.float32)), Buf.of(b.astype(np.float32)), Buf(16 * 16 * 4)
    H.launch(f, (1, 1, 1), (32, 1, 1), 0, [ctypes.c_uint64(ab.ptr), ctypes.c_uint64(bb.ptr), ctypes.c_uint64(db.ptr)])
    H.sync()
    return db.download((16, 16), np.float32).astype(np.float64)


rng = np.random.default_rng(3)
a = rng.standard_normal((16, 16)).astype(np.float32)
b = rng.standard_normal((16, 16)).astype(np.float32)
import refs as R
a16, b16 = R.bf16(a.astype(np.float64)), R.bf16(b.astype(np.float64))
ref = a16 @ b16
d = run(a16, b16)
m = C.cmp_bound(d, ref, C.gamma(16) * (np.abs(a16) @ np.abs(b16)) + 2.0 ** -24 * np.abs(ref))
C.record(kernel="wmma_f32_16x16x16_bf16", case="layout (random bf16): D = A B within gamma_16", **m)
pos = np.tile(np.arange(16, dtype=np.float64), (16, 1))
eye = np.eye(16)
C.record(kernel="wmma_f32_16x16x16_bf16", case="I @ B exact for B >= 0", ok=bool(np.array_equal(run(eye, pos), pos)))
d = run(eye, -pos)
C.record(kernel="wmma_f32_16x16x16_bf16", case="I @ B for B < 0 (rounding probe)", ok=None,
         exact=bool(np.array_equal(d, -pos)), max_ulps_off=float(np.max(np.abs(d + pos) / np.maximum(np.spacing(np.abs(pos).astype(np.float32)), 1e-45))))
sys.exit(C.done())
