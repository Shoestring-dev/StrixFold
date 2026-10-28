"""The kit's strict-mode coverage (no GPU work): every recorded and derived launch (common.launches(): the recording,
the prompt GEMM shapes from the weights, every decode / verify window size) has its own variant in the kit, chosen by
hiprt.Kit.find's exact rule (zig/src/gpu/hip/aot.zig's strict mode, TF_GPU_STRICT=1). aot_build.py skips a variant
that fails to compile and goes on, so a short kit shows here rather than as a fallback or a refusal in a run."""

import sys

import common as C
import hiprt as H
from aot_build import TORCH_TO_TRITON

C.log_open("t_cover")


def launch_of(case):
    """A launch with the case's specialization: pointers 16-byte aligned or not as recorded, its scalars, its
    constexprs (an int argument of 1 recorded as a constexpr stays one)."""
    args = {}
    for name, (dtype, _shape, _strides, aligned) in case["tensors"].items():
        args[name] = H.Ptr(0x10000 if aligned else 0x10008, TORCH_TO_TRITON[dtype])
    for name, v in case["scalars"].items():
        args[name] = float(v["float"]) if isinstance(v, dict) else int(v)
    consts = {n: (float(v["float"]) if isinstance(v, dict) else int(v)) for n, v in case["consts"].items()}
    return args, consts


def main():
    for fn, cases in sorted(C.launches().items()):
        missing = []
        for ci, c in enumerate(cases):
            args, consts = launch_of(c)
            try:
                C.KIT.find(fn, args, consts, exact_only=True)
            except KeyError:
                missing.append(f"{ci} {c['phase']}")
        C.record(kernel=fn, case="every launch has its own variant (strict)", ok=not missing, launches=len(cases),
                 missing=missing[:20])
    sys.exit(C.done())


if __name__ == "__main__":
    main()
