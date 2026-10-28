"""A minimal HIP runtime binding (ctypes on libamdhip64) and an AOT kit loader: what the native engine does with the
kit, without torch. Buffers, module loading, kernel launch with an argument buffer packed in the kernel's parameter
order, events for timing.

The kit loader (Kit) selects variants as zig/src/cuda/aot.zig's Set.find does (exact match on constexprs, parameter
types, 16-divisibility; an int argument of 1 must be a folded constexpr), and launches them with the Triton AMD ABI:
runtime parameters in order, then the global and profile scratch pointers, block = num_warps * warp_size.
"""

import ctypes
import json
import os
from pathlib import Path

import numpy as np

_lib = ctypes.CDLL(os.environ.get("TF_HIP_LIB", "/opt/rocm/lib/libamdhip64.so"))
_lib.hipGetErrorString.restype = ctypes.c_char_p

c_void_p = ctypes.c_void_p


def check(err, what=""):
    if err != 0:
        raise RuntimeError(f"HIP error {err} ({_lib.hipGetErrorString(err).decode()}) in {what}")


check(_lib.hipInit(0), "hipInit")
check(_lib.hipSetDevice(0), "hipSetDevice")


def device_name():
    class Props(ctypes.Structure):
        _fields_ = [("name", ctypes.c_char * 256), ("pad", ctypes.c_char * 4096)]
    # hipDeviceGetAttribute is enough for what tests record; the arch name comes from hipDeviceProp_t.gcnArchName,
    # whose offset varies by version, so use hipDeviceGetName for the name only
    buf = ctypes.create_string_buffer(256)
    check(_lib.hipDeviceGetName(buf, 256, 0), "hipDeviceGetName")
    return buf.value.decode()


class Buf:
    """A device allocation (hipMalloc) of nbytes; freed on del."""

    def __init__(self, nbytes):
        self.nbytes = int(max(nbytes, 1))
        p = c_void_p()
        check(_lib.hipMalloc(ctypes.byref(p), ctypes.c_size_t(self.nbytes)), f"hipMalloc({self.nbytes})")
        self.ptr = p.value

    @staticmethod
    def of(arr):
        arr = np.ascontiguousarray(arr)
        b = Buf(arr.nbytes)
        b.upload(arr)
        return b

    def upload(self, arr, offset=0):
        arr = np.ascontiguousarray(arr)
        assert offset + arr.nbytes <= self.nbytes
        check(_lib.hipMemcpy(c_void_p(self.ptr + offset), arr.ctypes.data_as(c_void_p), ctypes.c_size_t(arr.nbytes), 1),
              "hipMemcpy H2D")
        sync()      # a "blocking" copy may return before it completes (w-arch, HIP 7.15): order it explicitly

    def download(self, shape, dtype, offset=0):
        out = np.empty(shape, dtype=dtype)
        assert offset + out.nbytes <= self.nbytes
        sync()
        check(_lib.hipMemcpy(out.ctypes.data_as(c_void_p), c_void_p(self.ptr + offset), ctypes.c_size_t(out.nbytes), 2),
              "hipMemcpy D2H")
        return out

    def fill(self, byte):
        check(_lib.hipMemset(c_void_p(self.ptr), ctypes.c_int(byte), ctypes.c_size_t(self.nbytes)), "hipMemset")
        sync()

    def __del__(self):
        try:
            if self.ptr:
                _lib.hipFree(c_void_p(self.ptr))
                self.ptr = None
        except Exception:  # noqa: BLE001 - interpreter teardown
            pass


def sync():
    check(_lib.hipDeviceSynchronize(), "hipDeviceSynchronize")


class Module:
    def __init__(self, image: bytes):
        self._image = ctypes.create_string_buffer(image, len(image))
        m = c_void_p()
        check(_lib.hipModuleLoadData(ctypes.byref(m), self._image), "hipModuleLoadData")
        self.handle = m

    def function(self, name):
        f = c_void_p()
        check(_lib.hipModuleGetFunction(ctypes.byref(f), self.handle, name.encode()), f"hipModuleGetFunction({name})")
        return f


def launch(func, grid, block, shared, packed: list):
    """packed: ctypes values in parameter order (c_uint64 for pointers, c_int32, c_float, ...)."""
    holders = [ctypes.pointer(v) for v in packed]
    arr = (c_void_p * max(len(holders), 1))(*[ctypes.cast(h, c_void_p) for h in holders])
    check(_lib.hipModuleLaunchKernel(func, ctypes.c_uint(grid[0]), ctypes.c_uint(grid[1]), ctypes.c_uint(grid[2]),
                                     ctypes.c_uint(block[0]), ctypes.c_uint(block[1]), ctypes.c_uint(block[2]),
                                     ctypes.c_uint(shared), c_void_p(None), arr, c_void_p(None)),
          "hipModuleLaunchKernel")


class Event:
    def __init__(self):
        e = c_void_p()
        check(_lib.hipEventCreate(ctypes.byref(e)), "hipEventCreate")
        self.h = e

    def record(self):
        check(_lib.hipEventRecord(self.h, c_void_p(None)), "hipEventRecord")

    def ms_since(self, start):
        check(_lib.hipEventSynchronize(self.h), "hipEventSynchronize")
        t = ctypes.c_float()
        check(_lib.hipEventElapsedTime(ctypes.byref(t), start.h, self.h), "hipEventElapsedTime")
        return t.value


def time_ms(fn, iters=50, warmup=5):
    """Median-free mean device time of fn() over iters launches (events around the batch)."""
    for _ in range(warmup):
        fn()
    sync()
    a, b = Event(), Event()
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    return b.ms_since(a) / iters


# ---- the AOT kit ------------------------------------------------------------------------------------------------
class Ptr:
    """A pointer argument: its address and Triton type ("*bf16", ...); holds its buffer alive (an inline Buf.of(...)
    would otherwise be freed before the launch runs)."""

    def __init__(self, addr, ty, buf=None):
        self.addr, self.ty, self.buf = int(addr), ty, buf


def P(buf, ty, offset=0):
    return Ptr(buf.ptr + offset, ty, buf)


class Kit:
    def __init__(self, path):
        self.dir = Path(path)
        self.spec = json.loads((self.dir / "aot.json").read_text())
        self.kernels = self.spec["kernels"]
        self._loaded = {}
        self._scratch = None

    def _load(self, k):
        key = k["hash"]
        if key not in self._loaded:
            m = Module((self.dir / k["file"]).read_bytes())
            self._loaded[key] = (m, m.function(k["name"]))
        return self._loaded[key][1]

    @staticmethod
    def _matches(k, args, consts):
        for name, v in consts.items():
            got = k["consts"].get(name)
            if got is None:
                return False
            if isinstance(v, float):
                if "f32" not in got or got["f32"] != int(np.float32(v).view(np.uint32)):
                    return False
            elif "int" not in got or got["int"] != int(v):
                return False
        # every constexpr of the variant is given, or is an int argument of 1 that Triton folded
        for n in set(k["consts"]) - set(consts):
            a = args.get(n)
            if not (isinstance(a, (int, np.integer)) and a == 1 and k["consts"][n].get("int") == 1):
                return False
        runtime = 0
        params = {p["name"]: p for p in k["params"]}
        for name, a in args.items():
            p = params.get(name)
            if isinstance(a, Ptr):
                if p is None or p["type"] != a.ty or p["div16"] != (a.addr % 16 == 0):
                    return False
            elif isinstance(a, float):
                if p is None or p["type"] != "fp32":
                    return False
            elif isinstance(a, (int, np.integer)):
                if p is None:
                    if a != 1 or k["consts"].get(name, {}).get("int") != 1:
                        return False
                    continue
                if p["type"] != "i32" or (not p["nospec"] and a == 1):
                    return False
                if p["div16"] != ((not p["nospec"]) and a % 16 == 0):
                    return False
            else:
                raise TypeError(f"argument {name}: {type(a)}")
            runtime += 1
        return runtime == len(k["params"])

    @staticmethod
    def _fits(k, args, consts):
        """aot.zig's fits: the same constexprs, every assumption the variant makes held by the launch (an argument it
        takes as 16-divisible is; an int it folded to 1 is 1). None if not, else how many specializations it shares."""
        for name, v in consts.items():
            got = k["consts"].get(name)
            if got is None:
                return None
            if isinstance(v, float):
                if got.get("f32") != int(np.float32(v).view(np.uint32)):
                    return None
            elif got.get("int") != int(v):
                return None
        params = {p["name"]: p for p in k["params"]}
        held = runtime = 0
        for name, a in args.items():
            p = params.get(name)
            if isinstance(a, Ptr):
                if p is None or p["type"] != a.ty:
                    return None
                d16 = a.addr % 16 == 0
            elif isinstance(a, float):
                if p is None or p["type"] != "fp32":
                    return None
                held += 1
                runtime += 1
                continue
            else:
                if p is None:
                    if a != 1 or k["consts"].get(name, {}).get("int") != 1:
                        return None
                    held += 1
                    continue
                if p["type"] != "i32":
                    return None
                d16 = (not p["nospec"]) and a != 1 and a % 16 == 0
            if p["div16"] and not d16:
                return None
            held += p["div16"] == d16
            runtime += 1
        for n in set(k["consts"]) - set(consts):
            a = args.get(n)
            if not (isinstance(a, (int, np.integer)) and a == 1 and k["consts"][n].get("int") == 1):
                return None
        return held if runtime == len(k["params"]) else None

    def find(self, fn, args, consts, exact_only=False):
        for k in self.kernels:
            if k["fn"] == fn and self._matches(k, args, consts):
                return k
        if not exact_only:
            best, best_held = None, -1
            for k in self.kernels:
                if k["fn"] != fn:
                    continue
                h = self._fits(k, args, consts)
                if h is not None and h > best_held:
                    best, best_held = k, h
            if best is not None:
                return best
        raise KeyError(f"no variant of {fn} for consts {consts} args "
                       f"{ {n: (a.ty, a.addr % 16 == 0) if isinstance(a, Ptr) else a for n, a in args.items()} }")

    def run(self, fn, grid, args: dict, consts: dict, variant=None):
        k = variant or self.find(fn, args, consts)
        f = self._load(k)
        packed = []
        for p in k["params"]:
            a = args[p["name"]]
            if p["type"].startswith("*"):
                packed.append(ctypes.c_uint64(a.addr))
            elif p["type"] == "i32":
                packed.append(ctypes.c_int32(int(a)))
            elif p["type"] == "fp32":
                packed.append(ctypes.c_float(float(a)))
            else:
                raise TypeError(p["type"])
        if k["global_scratch"] or k["profile_scratch"]:
            raise RuntimeError(f"{fn}: variant needs scratch ({k['global_scratch']}, {k['profile_scratch']})")
        packed += [ctypes.c_uint64(0), ctypes.c_uint64(0)]
        launch(f, grid, (k["block"], 1, 1), k["shared"], packed)
        return k


# ---- bf16 / fp16 helpers ----------------------------------------------------------------------------------------
def f32_to_bf16_bits(x):
    """fp32 -> bf16 bits, round to nearest even (NaN stays NaN)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u >> 16) & 1) + 0x7FFF
    out = ((u + r) >> 16).astype(np.uint16)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    out[nan] = 0x7FC0
    return out


def bf16_bits_to_f32(b):
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_round(x):
    """x (any float) -> nearest bf16 value (via fp32 first when x is fp64: the reference rounds once from fp64
    through bf16_round64 instead)."""
    return bf16_bits_to_f32(f32_to_bf16_bits(np.asarray(x, dtype=np.float32)))


def bf16_round64(x):
    """fp64 -> nearest bf16 (round to nearest even, directly from fp64: no double rounding through fp32)."""
    import ml_dtypes
    return np.asarray(x, dtype=np.float64).astype(ml_dtypes.bfloat16).astype(np.float64)
