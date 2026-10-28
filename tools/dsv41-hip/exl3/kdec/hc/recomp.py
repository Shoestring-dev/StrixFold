"""recomp.py FILE...: recompile kit gfx1151-v5 Triton variants (by hsaco file name) with ~/venvs/triton, check the
hsaco is byte-identical to the kit's, and dump ttgir / llir / amdgcn beside (k-dec2, hc fusion stage 1)."""
import hashlib, inspect, json, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "tfp/tri/src"))
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import tri_kernels as K

KIT = Path.home() / "tfp/tri/kit/gfx1151-v5"
OUT = Path.home() / "tfp/kdec/hc/ir"
OUT.mkdir(parents=True, exist_ok=True)
m = json.load(open(KIT / "aot.json"))
want = set(sys.argv[1:])
done = set()
for k in m["kernels"]:
    f = Path(k["file"]).name
    if f not in want or f in done:
        continue
    done.add(f)
    fn = getattr(K, k["fn"])
    defaults = {n: p.default for n, p in inspect.signature(fn.fn).parameters.items() if p.default is not inspect.Parameter.empty}
    ptypes = {p["name"]: p for p in k["params"]}
    sig, constexprs, attrs = {}, {}, {}
    for i, name in enumerate(fn.arg_names):
        if name in k["consts"]:
            v = k["consts"][name]
            sig[name] = "constexpr"
            constexprs[name] = v["int"] if "int" in v else __import__("struct").unpack("<f", int(v["f32"]).to_bytes(4, "little"))[0]
        elif name in ptypes:
            sig[name] = ptypes[name]["type"]
            if ptypes[name]["div16"]:
                attrs[(i,)] = [["tt.divisibility", 16]]
        elif name in defaults:
            sig[name] = "constexpr"
            constexprs[name] = defaults[name].value if hasattr(defaults[name], "value") else defaults[name]
        else:  # an int folded to 1 by the recording
            sig[name] = "constexpr"
            constexprs[name] = 1
    opts = dict(k.get("options") or {"num_warps": k["num_warps"]})
    src = ASTSource(fn, sig, constexprs=constexprs, attrs=attrs)
    c = triton.compile(src, target=GPUTarget("hip", "gfx1151", 32), options=opts)
    sha = hashlib.sha256(c.asm["hsaco"]).hexdigest()
    same = sha == f[:-6]
    print(k["fn"], f[:12], "recompiled", sha[:12], "IDENTICAL" if same else "DIFFERENT")
    for ext in ("ttgir", "llir", "amdgcn"):
        (OUT / f"{k['fn']}-{f[:12]}.{ext}").write_text(c.asm[ext])
