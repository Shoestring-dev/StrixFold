"""Generate tri_kernels.py: the @triton.jit kernels of TensorFold's DeepSeek-V4.1 family (kernels.py, markov.py at
bertholomus/TensorFold 222a9c0b, Apache-2.0) and exl3/prefill.py's _gemm, made portable for AMD gfx1151.

The kernel bodies are copied verbatim; only the NVIDIA-only constructs and one numerical defect are replaced, by name,
and every replacement is listed in PORT_EDITS (written into the generated header and checked: each edit must apply
exactly the stated number of times, so an upstream change cannot slip through silently).

  - gdc_wait / gdc_launch_dependents (PDL, griddepcontrol): no-op functions. The AMD build compiles every variant with
    PDL=False; ordering comes from the HIP stream (each kernel starts after the previous one finished).
  - _prefetch_l2 (inline PTX prefetch.global.L2) and _discard_l2 (discard.global.L2): no-ops. Neither moves data a
    kernel reads, so results are unchanged; only the cache hint is lost.
  - float8e4nv conversions (cvt.rn.satfinite.e4m3x2.f32 and the e4m3 -> f32 widening): integer-op implementations
    _e4m3_rt / _e4m3_decode (round to nearest even, values already clamped to +-448 by the callers).
  - The router score (_route_row, _route_parts): upstream's sqrt(log(1 + exp(lg))) in fp32 becomes _sqrt_softplus
    (F.softplus's formula, model.py's Gate). Upstream's 1 + e rounds to 1 below lg = -16.6, so every expert there
    scored only its bias (first light's 12 routing violations), and costs 2^-24 / e relative above it; on gfx1151
    tl.exp adds ~|lg| u32 more. A correctness fix, not a port: the CUDA build has the same defect.

Usage: python make_port.py UPSTREAM_DIR OUT_FILE
"""

import ast
import hashlib
import sys
from pathlib import Path

SOURCES = ["kernels.py", "markov.py", "prefill.py"]

# (file, old, new, count): exact substring edits applied to the copied function sources
PORT_EDITS = [
    ("kernels.py",
     "re = (tl.minimum(tl.maximum(re / sb, -448.0), 448.0)).to(tl.float8e4nv).to(tl.float32) * sb",
     "re = _e4m3_rt(tl.minimum(tl.maximum(re / sb, -448.0), 448.0)) * sb", 2),
    ("kernels.py",
     "im = (tl.minimum(tl.maximum(im / sb, -448.0), 448.0)).to(tl.float8e4nv).to(tl.float32) * sb",
     "im = _e4m3_rt(tl.minimum(tl.maximum(im / sb, -448.0), 448.0)) * sb", 2),
    ("kernels.py", "sl = sl.to(tl.float8e4nv, bitcast=True).to(tl.float32)", "sl = _e4m3_decode(sl)", 1),
    ("kernels.py", "sh = sh.to(tl.float8e4nv, bitcast=True).to(tl.float32)", "sh = _e4m3_decode(sh)", 1),
    ("kernels.py", "sc = sb.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)",
     "sc = _e4m3_decode(sb.to(tl.uint8))", 1),
    ("kernels.py", "sp = tl.where(lg > 20.0, lg, tl.log(1.0 + tl.exp(lg)))\n    sc = tl.sqrt(sp)",
     "sc = _sqrt_softplus(lg)", 2),
]

# NVIDIA-only helpers whose upstream definitions are dropped (replaced in PRELUDE)
DROPPED = {"_prefetch_l2", "_discard_l2"}

PRELUDE = '''
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def gdc_wait():
    """PDL wait (griddepcontrol.wait): a no-op on AMD; variants are compiled with PDL=False and stream order applies."""
    pass


@triton.jit
def gdc_launch_dependents():
    """PDL trigger (griddepcontrol.launch_dependents): a no-op on AMD."""
    pass


@triton.jit
def _prefetch_l2(ptrs):
    """prefetch.global.L2 in the CUDA build: a no-op here (it moves no data a kernel reads)."""
    pass


@triton.jit
def _discard_l2(ptrs):
    """discard.global.L2 in the CUDA build: a no-op here (the data is dead either way)."""
    pass


@triton.jit
def _e4m3_code_pos(f):
    """E4M3 (fn) code of f >= 0 (fp32, at most 448 after the callers' clamp), round to nearest even: subnormals
    through the 2^-9-step add (an IEEE fp32 add rounds to nearest even), normals by rebiasing the exponent and an RNE
    shift by 20 bits; 448 and above saturate to 0x7E (callers clamp first)."""
    fb = f.to(tl.int32, bitcast=True)
    sub = (f + 16384.0).to(tl.int32, bitcast=True) - (141 << 23)
    nor = (fb + (-120 << 23) + 0x7FFFF + ((fb >> 20) & 1)) >> 20
    nor = tl.where(nor >= 0x7F, 0x7E, nor)
    return tl.where(fb >= (1087 << 20), 0x7E, tl.where(fb < (121 << 23), sub, nor))


@triton.jit
def _e4m3_decode(code):
    """E4M3 (fn) byte (any integer dtype holding 0..255) -> fp32, exactly: S.EEEE.MMM, bias 7, subnormal step 2^-9,
    S.1111.111 NaN."""
    c = code.to(tl.int32) & 0xFF
    e = (c >> 3) & 15
    m = c & 7
    nor = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    sub = m.to(tl.float32) * 0.001953125
    v = tl.where(e == 0, sub, nor)
    v = tl.where((c & 0x7F) == 0x7F, float("nan"), v)
    return tl.where((c & 0x80) != 0, -v, v)


@triton.jit
def _e4m3_rt(x):
    """fp32 -> E4M3 (round to nearest even) -> fp32, x within [-448, 448]; the sign is kept (-0 stays -0), as the
    CUDA build's cvt.rn.satfinite.e4m3x2.f32 then widening gives."""
    bits = x.to(tl.int32, bitcast=True)
    a = (bits & 0x7FFFFFFF).to(tl.float32, bitcast=True)
    v = _e4m3_decode(_e4m3_code_pos(a))
    return (v.to(tl.int32, bitcast=True) | (bits & -2147483648)).to(tl.float32, bitcast=True)


@triton.jit
def _exp_rr(x):
    """e^x in fp32 within 1.4 u32 (measured over [-87, 30]). tl.exp lowers to exp2(fl(x log2 e)) here, and
    libdevice.exp is rewritten to the same: the product's rounding costs ~|x| u32 (64 u32 near -87). Cody-Waite instead:
    x = n ln2 + r, |r| <= ln2 / 2, r in two fma steps (ln2 = hi + lo, hi with 15 significant bits), e^r by its
    degree-7 Taylor polynomial (Horner, truncation < 0.2 u32), times 2^n by ldexp (exact; subnormal results rounded
    once). NaN stays NaN."""
    x = tl.clamp(x, -150.0, 100.0, propagate_nan=tl.PropagateNan.ALL)
    n = tl.floor(x * 1.4426950408889634 + 0.5)
    r = tl.fma(n, -0.693145751953125, x)
    r = tl.fma(n, -1.4286068203094172e-06, r)
    p = tl.fma(r, 1.0 / 5040.0, 1.0 / 720.0)
    p = tl.fma(p, r, 1.0 / 120.0)
    p = tl.fma(p, r, 1.0 / 24.0)
    p = tl.fma(p, r, 1.0 / 6.0)
    p = tl.fma(p, r, 0.5)
    p = tl.fma(p, r, 1.0)
    p = tl.fma(p, r, 1.0)
    return libdevice.ldexp(p, n.to(tl.int32))


@triton.jit
def _sqrt_softplus(lg):
    """The MoE router's score sqrt(softplus(lg)) in fp32 (model.py Gate: F.softplus(scores).sqrt()), within 4 u32 of
    the exact value for finite lg >= -174 (numerics.json router.noaux_tc: softplus/sqrt <= 4 ulp): lg itself above 20
    and log1p(exp(lg)) below, as F.softplus computes it (libdevice.log1p, _exp_rr), its square root correctly rounded;
    below lg = -16 the score is exp(lg / 2) directly (sqrt(log1p(e)) = e^(1/2) (1 - e/4 + ...) with e < 2^-23), which
    stays a normal number down to lg = -174 where exp(lg) is subnormal below -87."""
    sp = tl.where(lg > 20.0, lg, libdevice.log1p(_exp_rr(lg)))
    return tl.where(lg < -16.0, _exp_rr(0.5 * lg), tl.sqrt_rn(sp))

'''


def is_jit(fn: ast.FunctionDef) -> bool:
    for d in fn.decorator_list:
        s = ast.unparse(d)
        if s.startswith("triton.jit"):
            return True
    return False


def main(updir: str, out: str) -> None:
    up = Path(updir)
    parts = []
    hashes = []
    used = {k: 0 for k in range(len(PORT_EDITS))}
    names = []
    for src in SOURCES:
        text = (up / src).read_text()
        hashes.append((src, hashlib.sha256(text.encode()).hexdigest()))
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)
        chunk = [f"\n# ---- from {src} " + "-" * (100 - len(src)) + "\n"]
        for node in tree.body:
            if isinstance(node, ast.Assign) and src == "markov.py" and ast.unparse(node.targets[0]) == "BIG":
                chunk.append(ast.get_source_segment(text, node) + "\n")
            if not (isinstance(node, ast.FunctionDef) and is_jit(node)):
                continue
            if node.name in DROPPED:
                continue
            start = node.decorator_list[0].lineno - 1
            body = "".join(lines[start:node.end_lineno])
            for k, (f, old, new, n) in enumerate(PORT_EDITS):
                if f == src and old in body:
                    used[k] += body.count(old)
                    body = body.replace(old, new)
            for bad in ("float8e4nv", "inline_asm", "cache_modifier", "extra.cuda"):
                if bad in body and not (bad == "cache_modifier" and node.name == "_route_row"):
                    raise SystemExit(f"{src}:{node.name}: NVIDIA construct {bad!r} left unported")
            names.append(node.name)
            chunk.append("\n\n" + body)
        parts.append("".join(chunk))
    for k, (f, old, new, n) in enumerate(PORT_EDITS):
        if used[k] != n:
            raise SystemExit(f"edit {k} ({f}) applied {used[k]} times, expected {n}")
    head = ['"""AMD gfx1151 port of TensorFold\'s DeepSeek-V4.1 Triton kernels (generated by make_port.py; do not edit).',
            "",
            "Upstream: bertholomus/TensorFold @ 222a9c0b6ecaa9fd3c5dbded65a668524c9510ff (Apache-2.0, see LICENSE):",
            "src/tensorfold/families/deepseek_v41/cuda/{kernels,markov}.py, src/tensorfold/cuda/exl3/prefill.py.",
            "Source hashes (sha256):"]
    head += [f"  {s}: {h}" for s, h in hashes]
    head += ["Edits (besides the no-op PDL / L2-hint replacements in the prelude):"]
    head += [f"  {f}: {old!r} -> {new!r} (x{n})" for f, old, new, n in PORT_EDITS]
    head += ['"""', ""]
    Path(out).write_text("\n".join(head) + PRELUDE + "".join(parts))
    print(f"{out}: {len(names)} jit functions: {' '.join(names)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
