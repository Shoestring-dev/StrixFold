"""Bounded FP32 HIP attention for the optional BF16 vision worker."""

import ctypes as C
import hashlib
import os

HEADS, DIM, TILE = 16, 64, 128
pointer, integer, size, stride = C.c_void_p, C.c_int, C.c_size_t, C.c_int64
float_ptr = C.POINTER(C.c_float)


def load_vision_code(path, expected):
    with open(path, 'rb') as stream:
        code = stream.read((1 << 20) + 1)
    if not 0 < len(code) <= 1 << 20 or hashlib.sha256(code).hexdigest() != expected:
        raise ValueError('vision HIP code failed SHA256 or size check')
    return C.create_string_buffer(code)

def call(library, name, args, *values):
    fn = getattr(library, name)
    fn.argtypes = args
    fn.restype = integer
    status = fn(*values)
    if status:
        raise RuntimeError(f'{name} failed with status {status}')


def attention_required_bytes(tokens, reserve_mib):
    if reserve_mib < 0:
        raise ValueError('vision memory reserve must be nonnegative')
    workspace = HEADS * tokens * (8 * DIM + TILE) * 4
    return reserve_mib * 1024**2 + workspace


def hip_attention(q, k, v, image):
    n = q.shape[-2]
    with open('/proc/meminfo', encoding='ascii') as stream:
        available = next(int(line.split()[1]) * 1024 for line in stream if line.startswith('MemAvailable:'))
    reserve = int(os.environ.get('TF_DS_VISION_RESERVE_MIB', '0'))
    if available < attention_required_bytes(n, reserve):
        raise MemoryError('available RAM is below the image workspace and configured reserve')
    qf, kf, vf = (x.float().contiguous() for x in (q, k, v))
    output = torch.empty_like(qf)
    return _attention(n, qf, kf, vf, output, image).to(q.dtype)


def _attention(n, q, k, v, out, image):
    hip, blas = C.CDLL('libamdhip64.so'), C.CDLL('libhipblas.so')
    vector_bytes = HEADS * n * DIM * 4
    blocks = []
    handle, module, kernel = pointer(), pointer(), pointer()

    def allocate(nbytes):
        p = pointer()
        call(hip, 'hipMalloc', [C.POINTER(pointer), size], C.byref(p), nbytes)
        blocks.append(p)
        return p

    def gemm(trans_a, m, n, k, a, lda, stride_a, b, ldb, stride_b, c, ldc, stride_c, scale):
        alpha, beta = C.c_float(scale), C.c_float(0)
        types = [pointer, integer, integer, integer, integer, integer, float_ptr,
                 pointer, integer, stride, pointer, integer, stride, float_ptr,
                 pointer, integer, stride, integer]
        call(blas, 'hipblasSgemmStridedBatched', types, handle, trans_a, 111, m, n, k,
             C.byref(alpha), a, lda, stride_a, b, ldb, stride_b, C.byref(beta), c, ldc, stride_c, HEADS)

    call(hip, 'hipSetDevice', [integer], int(os.environ.get('TF_HIP_DEVICE', '0')))
    try:
        call(blas, 'hipblasCreate', [C.POINTER(pointer)], C.byref(handle))
        call(hip, 'hipModuleLoadData', [C.POINTER(pointer), pointer], C.byref(module), image)
        call(hip, 'hipModuleGetFunction', [C.POINTER(pointer), pointer, C.c_char_p],
             C.byref(kernel), module, b'normalize_scores')
        dq, dk, dv = [allocate(vector_bytes) for _ in range(3)]
        ds = allocate(HEADS * n * TILE * 4)
        dy = allocate(vector_bytes)
        for dst, src in zip((dq, dk, dv), (q, k, v)):
            call(hip, 'hipMemcpy', [pointer, pointer, size, integer], dst, src.data_ptr(), vector_bytes, 1)
        keys = integer(n)
        args = (pointer * 2)(C.cast(C.byref(ds), pointer), C.cast(C.byref(keys), pointer))
        for offset in range(0, n, TILE):
            q_rows = min(TILE, n - offset)
            # Row-major S=QK^T is column-major S^T=KQ^T.
            gemm(112, n, q_rows, DIM, dk, DIM, n * DIM, dq.value + offset * DIM * 4,
                 DIM, n * DIM, ds, n, n * q_rows, .125)
            call(hip, 'hipModuleLaunchKernel', [pointer] + [C.c_uint] * 7 + [pointer] * 3,
                 kernel, HEADS * q_rows, 1, 1, 256, 1, 1, 0, None, args, None)
            gemm(111, DIM, q_rows, n, dv, DIM, n * DIM, ds, n, n * q_rows,
                 dy.value + offset * DIM * 4, DIM, n * DIM, 1.0)
        call(hip, 'hipDeviceSynchronize', [])
        call(hip, 'hipMemcpy', [pointer, pointer, size, integer], out.data_ptr(), dy, vector_bytes, 2)
        return out
    finally:
        for p in blocks:
            call(hip, 'hipFree', [pointer], p)
        if module:
            call(hip, 'hipModuleUnload', [pointer], module)
        if handle:
            call(blas, 'hipblasDestroy', [pointer], handle)
