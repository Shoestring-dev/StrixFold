/* Compile-time agreement between hipblas.zig's hand-declared values and prototypes and the installed hipBLAS headers
 * (the same discipline as zig/src/gpu/hip/hip_abi_check.c). Nothing here runs. */
#include <hipblas/hipblas.h>

#define SAME_FN(fn, type) _Static_assert(__builtin_types_compatible_p(__typeof__(&fn), type), #fn " prototype")
#define VALUE(name, v) _Static_assert((long long)(name) == (long long)(v), #name " value")

VALUE(HIPBLAS_STATUS_SUCCESS, 0);
VALUE(HIPBLAS_OP_N, 111);
VALUE(HIPBLAS_OP_T, 112);
VALUE(HIPBLAS_POINTER_MODE_HOST, 0);
VALUE(HIPBLAS_ATOMICS_NOT_ALLOWED, 0);
_Static_assert(sizeof(hipblasOperation_t) == 4, "hipblasOperation_t");
_Static_assert(sizeof(hipblasStatus_t) == 4, "hipblasStatus_t");
_Static_assert(sizeof(hipblasPointerMode_t) == 4, "hipblasPointerMode_t");
_Static_assert(sizeof(hipblasAtomicsMode_t) == 4, "hipblasAtomicsMode_t");
VALUE(HIP_R_16BF, 14);
VALUE(HIP_R_32F, 0);
VALUE(HIPBLAS_COMPUTE_32F, 2);
VALUE(HIPBLAS_GEMM_DEFAULT, 160);
_Static_assert(sizeof(hipDataType) == 4, "hipDataType");
_Static_assert(sizeof(hipblasComputeType_t) == 4, "hipblasComputeType_t");
_Static_assert(sizeof(hipblasGemmAlgo_t) == 4, "hipblasGemmAlgo_t");

SAME_FN(hipblasCreate, hipblasStatus_t (*)(hipblasHandle_t *));
SAME_FN(hipblasDestroy, hipblasStatus_t (*)(hipblasHandle_t));
SAME_FN(hipblasSetStream, hipblasStatus_t (*)(hipblasHandle_t, hipStream_t));
SAME_FN(hipblasSetWorkspace, hipblasStatus_t (*)(hipblasHandle_t, void *, size_t));
SAME_FN(hipblasSetPointerMode, hipblasStatus_t (*)(hipblasHandle_t, hipblasPointerMode_t));
SAME_FN(hipblasSetAtomicsMode, hipblasStatus_t (*)(hipblasHandle_t, hipblasAtomicsMode_t));
SAME_FN(hipblasSgemm, hipblasStatus_t (*)(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                                          const float *, const float *, int, const float *, int, const float *, float *,
                                          int));

SAME_FN(hipblasGemmEx, hipblasStatus_t (*)(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                                           const void *, const void *, hipDataType, int, const void *, hipDataType, int,
                                           const void *, void *, hipDataType, int, hipblasComputeType_t,
                                           hipblasGemmAlgo_t));

/* Linked so the object is not dropped; the number of checked prototypes. */
int tf_hipblas_abi_checked(void) { return 8; }
