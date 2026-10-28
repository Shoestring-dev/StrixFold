#!/usr/bin/env python3
"""residency.py FILE...: each file's pages resident in the page cache (fincore's count; the boxes have no fincore), by
mincore(2) over a read-only shared mapping (nothing is read). One JSON line: {file: {bytes, pages, resident, fraction}}.
Standard library only."""
import ctypes
import json
import mmap
import os
import sys

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)
libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)


def resident(path):
    size = os.path.getsize(path)
    page = mmap.PAGESIZE
    pages = (size + page - 1) // page
    if size == 0:
        return {"bytes": 0, "pages": 0, "resident": 0, "fraction": 0.0}
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap " + path)
        try:
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore " + path)
            data = bytes(vec)
            n = sum(data.count(v) for v in range(1, 256, 2))  # bit 0: resident
        finally:
            libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)
    return {"bytes": size, "pages": pages, "resident": n, "fraction": round(n / pages, 6)}


print(json.dumps({p: resident(p) for p in sys.argv[1:]}))
