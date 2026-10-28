"""Experiment: which phase of aux_attn.hip's chunk loop costs the time (prompt, 2048 rows, one pass). Builds copies of
the kernel with one part compiled out and times each with HIP events (results are not checked: timing only).
Caveat: when a knockout leaves an LDS buffer unread, the compiler also drops the code that writes it (and the loads
feeding it), so a knockout can remove more than its own phase; x_attn_ab.py (whole builds, interleaved) is the
reliable comparison. "build" as the only argument compiles the variants (run that under the CPU limits)."""
import ctypes, os, re, subprocess, sys
from pathlib import Path
import numpy as np
import common as C, hiprt as H
import x_hipattn_perf as XP
import t_hipattn as T

SRC = Path.home() / "tfp/wt/tri/zig/kernels/hip/dsv41/aux_attn.hip"
INC = SRC.parent
OUT = C.ROOT / "build" / "phases"
OUT.mkdir(parents=True, exist_ok=True)
MARK = {"softmax": "// 2. the softmax step", "pv": "// 3. acc ="}


def without(text, mark):
    i = text.index(mark)
    j = text.index("{", i)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[:j] + "{ }" + text[k + 1:]


def build(name, text):
    src = OUT / f"{name}.hip"
    obj = OUT / f"{name}.hsaco"
    if obj.exists() and src.exists() and src.read_text() == text:
        return obj                                  # built by "x_attn_phases.py build" (run that under the CPU limits)
    src.write_text(text)
    subprocess.run(["/usr/bin/hipcc", "--offload-arch=gfx1151", "--offload-device-only", "--no-gpu-bundle-output", "-O3",
                    "-std=c++20", "-fno-gpu-rdc", "-mno-wavefrontsize64", "-ffp-contract=off", "-fno-fast-math",
                    "-fdenormal-fp-math=ieee", f"-I{INC}", str(src), "-o", str(obj)], check=True,
                   env={k: v for k, v in os.environ.items() if not k.startswith("HIPCC_")})
    return obj


if __name__ == "__main__":
    base = SRC.read_text()
    variants = {"full": base}
    for k, m in MARK.items():
        variants["no_" + k] = without(base, m)
    # finer: no transposed (kt) stores; no global key loads (registers zero); no FP4 decode (raw words as bf16)
    v = base.replace("                kr[sw] = tf_u32x4{col[0], col[1], col[2], col[3]};\n                kr[sw ^ 1] = tf_u32x4{col[4], col[5], col[6], col[7]};",
                     "                if (col[0] == 0x12345u) kr[sw] = tf_u32x4{col[0], col[1], col[2], col[3]};")
    assert v != base
    variants["no_kt_store"] = v
    v = base.replace("if (c + 2 * splits < nchunks) load(c + 2 * splits, nxt);", "nxt.kind = c < 4 ? 0 : 3;")
    assert v != base
    variants["no_global_load"] = v
    v = base.replace("} else if (cur.kind == 3) {", "} else if (cur.kind == 99) {")
    assert v != base
    variants["no_fp4_decode"] = v
    objs = {n: build(n, t) for n, t in variants.items()}
    if sys.argv[1:] == ["build"]:
        sys.exit(0)
    C.gpu_lock()
    d = XP.inputs(2048, True)
    keep = []
    # one set of device buffers; the launch alone is timed (hip_attn's argument struct captured from a first call)
    captured = {}
    real_launch = H.launch
    def grab(f, grid, block, shared, args):
        captured["a"] = args[0]
        captured["grid"] = grid
        return real_launch(f, grid, block, shared, args)
    H.launch = grab
    T.hip_attn(d, 1, keep)
    H.launch = real_launch
    a, grid = captured["a"], captured["grid"]
    print("load", C.quiet_window("w-tri-phases"), flush=True)
    for n, o in objs.items():
        f = H.Module(o.read_bytes()).function("tf_ds_aux_sparse_attn_h32_d512")
        ms = H.time_ms(lambda: H.launch(f, grid, (T.BLOCK, 1, 1), 0, [a]), iters=5, warmup=1)
        print(f"{n:12s} {ms:8.2f} ms", flush=True)
