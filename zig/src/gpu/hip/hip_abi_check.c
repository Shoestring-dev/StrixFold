/* Compile-time agreement between abi.zig's hand-declared HIP values and prototypes and the installed HIP headers.
 * Built into every HIP module; nothing here runs. A header change that alters a value, a layout or a prototype the
 * backend relies on fails the build here instead of corrupting a call at run time. Pointer constness is ABI-neutral;
 * every width and every by-value type is checked. */
#include <hip/hip_runtime_api.h>
#include <stddef.h>

#define SAME_FN(fn, type) _Static_assert(__builtin_types_compatible_p(__typeof__(&fn), type), #fn " prototype")
#define VALUE(name, v) _Static_assert((long long)(name) == (long long)(v), #name " value")

/* results */
VALUE(hipSuccess, 0);
VALUE(hipErrorOutOfMemory, 2);
VALUE(hipErrorInvalidDeviceFunction, 98);
VALUE(hipErrorInvalidImage, 200);
VALUE(hipErrorNoBinaryForGpu, 209);
VALUE(hipErrorNotFound, 500);
VALUE(hipErrorNotReady, 600);
VALUE(hipErrorECCNotCorrectable, 214);
VALUE(hipErrorIllegalAddress, 700);
VALUE(hipErrorLaunchTimeOut, 702);
VALUE(hipErrorAssert, 710);
VALUE(hipErrorLaunchFailure, 719);
VALUE(hipErrorStreamCaptureUnsupported, 900);

/* flags */
VALUE(hipStreamNonBlocking, 1);
VALUE(hipEventDisableTiming, 2);
VALUE(hipHostMallocPortable, 0x1);
VALUE(hipHostMallocMapped, 0x2);
VALUE(hipHostMallocWriteCombined, 0x4);
VALUE(hipHostMallocCoherent, 0x40000000);
VALUE(hipHostMallocNonCoherent, 0x80000000u);
VALUE(hipHostRegisterPortable, 0x1);
VALUE(hipHostRegisterMapped, 0x2);
VALUE(hipDeviceMallocDefault, 0x0);
VALUE(hipDeviceMallocFinegrained, 0x1);
VALUE(hipDeviceMallocUncached, 0x3);

/* enums */
VALUE(hipStreamCaptureModeGlobal, 0);
VALUE(hipStreamCaptureModeThreadLocal, 1);
VALUE(hipStreamCaptureModeRelaxed, 2);
VALUE(hipStreamCaptureStatusNone, 0);
VALUE(hipStreamCaptureStatusActive, 1);
VALUE(hipStreamCaptureStatusInvalidated, 2);
VALUE(hipDeviceAttributeClockRate, 5);
VALUE(hipDeviceAttributeCooperativeLaunch, 10);
VALUE(hipDeviceAttributeIntegrated, 16);
VALUE(hipDeviceAttributeL2CacheSize, 19);
VALUE(hipDeviceAttributeMaxThreadsPerBlock, 56);
VALUE(hipDeviceAttributeMultiprocessorCount, 63);
VALUE(hipDeviceAttributeMaxSharedMemoryPerBlock, 74);
VALUE(hipDeviceAttributeSharedMemPerBlockOptin, 75);
VALUE(hipDeviceAttributeWarpSize, 87);
VALUE(hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, 10002);
VALUE(hipDeviceAttributeWallClockRate, 10017);
VALUE(HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0);
VALUE(HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, 1);
VALUE(HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, 2);
VALUE(HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, 3);
VALUE(HIP_FUNC_ATTRIBUTE_NUM_REGS, 4);
VALUE(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 8);
VALUE(hipGraphExecUpdateSuccess, 0);
VALUE(hipGraphExecUpdateError, 1);
VALUE(hipGraphExecUpdateErrorTopologyChanged, 2);
VALUE(hipGraphExecUpdateErrorNodeTypeChanged, 3);
VALUE(hipGraphExecUpdateErrorFunctionChanged, 4);
VALUE(hipGraphExecUpdateErrorParametersChanged, 5);
VALUE(hipGraphExecUpdateErrorNotSupported, 6);
VALUE(hipGraphExecUpdateErrorUnsupportedFunctionChange, 7);

/* widths of what abi.zig passes by value */
_Static_assert(sizeof(hipError_t) == 4, "hipError_t");
_Static_assert(sizeof(hipDeviceAttribute_t) == 4, "hipDeviceAttribute_t");
_Static_assert(sizeof(hipStreamCaptureMode) == 4, "hipStreamCaptureMode");
_Static_assert(sizeof(hipStreamCaptureStatus) == 4, "hipStreamCaptureStatus");
_Static_assert(sizeof(hipGraphExecUpdateResult) == 4, "hipGraphExecUpdateResult");
_Static_assert(sizeof(hipFunction_attribute) == 4, "hipFunction_attribute");
_Static_assert(sizeof(hipDeviceptr_t) == 8, "hipDeviceptr_t");

/* hipKernelNodeParams, field by field (abi.zig KernelNodeParams) */
_Static_assert(sizeof(hipKernelNodeParams) == 64, "hipKernelNodeParams size");
_Static_assert(offsetof(hipKernelNodeParams, blockDim) == 0, "blockDim");
_Static_assert(offsetof(hipKernelNodeParams, extra) == 16, "extra");
_Static_assert(offsetof(hipKernelNodeParams, func) == 24, "func");
_Static_assert(offsetof(hipKernelNodeParams, gridDim) == 32, "gridDim");
_Static_assert(offsetof(hipKernelNodeParams, kernelParams) == 48, "kernelParams");
_Static_assert(offsetof(hipKernelNodeParams, sharedMemBytes) == 56, "sharedMemBytes");
_Static_assert(sizeof(dim3) == 12, "dim3");

/* prototypes, in abi.zig's Api order */
SAME_FN(hipInit, hipError_t (*)(unsigned int));
SAME_FN(hipRuntimeGetVersion, hipError_t (*)(int *));
SAME_FN(hipGetDeviceCount, hipError_t (*)(int *));
SAME_FN(hipSetDevice, hipError_t (*)(int));
SAME_FN(hipGetDevice, hipError_t (*)(int *));
SAME_FN(hipDeviceGetName, hipError_t (*)(char *, int, hipDevice_t));
SAME_FN(hipDeviceGetAttribute, hipError_t (*)(int *, hipDeviceAttribute_t, int));
SAME_FN(hipDeviceSynchronize, hipError_t (*)(void));
SAME_FN(hipMemGetInfo, hipError_t (*)(size_t *, size_t *));
SAME_FN(hipGetErrorName, const char *(*)(hipError_t));
SAME_FN(hipGetErrorString, const char *(*)(hipError_t));
SAME_FN(hipMalloc, hipError_t (*)(void **, size_t));
SAME_FN(hipExtMallocWithFlags, hipError_t (*)(void **, size_t, unsigned int));
SAME_FN(hipFree, hipError_t (*)(void *));
SAME_FN(hipHostMalloc, hipError_t (*)(void **, size_t, unsigned int));
SAME_FN(hipHostFree, hipError_t (*)(void *));
SAME_FN(hipHostRegister, hipError_t (*)(void *, size_t, unsigned int));
SAME_FN(hipHostUnregister, hipError_t (*)(void *));
SAME_FN(hipHostGetDevicePointer, hipError_t (*)(void **, void *, unsigned int));
SAME_FN(hipMemcpyHtoD, hipError_t (*)(hipDeviceptr_t, const void *, size_t));
SAME_FN(hipMemcpyDtoH, hipError_t (*)(void *, hipDeviceptr_t, size_t));
SAME_FN(hipMemcpyDtoD, hipError_t (*)(hipDeviceptr_t, hipDeviceptr_t, size_t));
SAME_FN(hipMemcpyHtoDAsync, hipError_t (*)(hipDeviceptr_t, const void *, size_t, hipStream_t));
SAME_FN(hipMemcpyDtoHAsync, hipError_t (*)(void *, hipDeviceptr_t, size_t, hipStream_t));
SAME_FN(hipMemcpyDtoDAsync, hipError_t (*)(hipDeviceptr_t, hipDeviceptr_t, size_t, hipStream_t));
SAME_FN(hipMemsetD8, hipError_t (*)(hipDeviceptr_t, unsigned char, size_t));
SAME_FN(hipMemsetD8Async, hipError_t (*)(hipDeviceptr_t, unsigned char, size_t, hipStream_t));
SAME_FN(hipMemsetD32, hipError_t (*)(hipDeviceptr_t, int, size_t));
SAME_FN(hipMemsetD32Async, hipError_t (*)(hipDeviceptr_t, int, size_t, hipStream_t));
SAME_FN(hipStreamCreateWithFlags, hipError_t (*)(hipStream_t *, unsigned int));
SAME_FN(hipStreamDestroy, hipError_t (*)(hipStream_t));
SAME_FN(hipStreamSynchronize, hipError_t (*)(hipStream_t));
SAME_FN(hipStreamQuery, hipError_t (*)(hipStream_t));
SAME_FN(hipStreamWaitEvent, hipError_t (*)(hipStream_t, hipEvent_t, unsigned int));
SAME_FN(hipStreamIsCapturing, hipError_t (*)(hipStream_t, hipStreamCaptureStatus *));
SAME_FN(hipEventCreateWithFlags, hipError_t (*)(hipEvent_t *, unsigned int));
SAME_FN(hipEventDestroy, hipError_t (*)(hipEvent_t));
SAME_FN(hipEventRecord, hipError_t (*)(hipEvent_t, hipStream_t));
SAME_FN(hipEventSynchronize, hipError_t (*)(hipEvent_t));
SAME_FN(hipEventQuery, hipError_t (*)(hipEvent_t));
SAME_FN(hipEventElapsedTime, hipError_t (*)(float *, hipEvent_t, hipEvent_t));
SAME_FN(hipModuleLoadData, hipError_t (*)(hipModule_t *, const void *));
SAME_FN(hipModuleUnload, hipError_t (*)(hipModule_t));
SAME_FN(hipModuleGetFunction, hipError_t (*)(hipFunction_t *, hipModule_t, const char *));
SAME_FN(hipModuleGetGlobal, hipError_t (*)(hipDeviceptr_t *, size_t *, hipModule_t, const char *));
SAME_FN(hipFuncGetAttribute, hipError_t (*)(int *, hipFunction_attribute, hipFunction_t));
SAME_FN(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, hipError_t (*)(int *, hipFunction_t, int, size_t));
SAME_FN(hipModuleLaunchKernel, hipError_t (*)(hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int,
                                              unsigned int, unsigned int, unsigned int, hipStream_t, void **, void **));
SAME_FN(hipStreamBeginCapture, hipError_t (*)(hipStream_t, hipStreamCaptureMode));
SAME_FN(hipStreamEndCapture, hipError_t (*)(hipStream_t, hipGraph_t *));
SAME_FN(hipGraphCreate, hipError_t (*)(hipGraph_t *, unsigned int));
SAME_FN(hipGraphDestroy, hipError_t (*)(hipGraph_t));
SAME_FN(hipGraphAddKernelNode, hipError_t (*)(hipGraphNode_t *, hipGraph_t, const hipGraphNode_t *, size_t,
                                              const hipKernelNodeParams *));
SAME_FN(hipGraphKernelNodeSetParams, hipError_t (*)(hipGraphNode_t, const hipKernelNodeParams *));
SAME_FN(hipGraphAddDependencies, hipError_t (*)(hipGraph_t, const hipGraphNode_t *, const hipGraphNode_t *, size_t));
SAME_FN(hipGraphGetNodes, hipError_t (*)(hipGraph_t, hipGraphNode_t *, size_t *));
SAME_FN(hipGraphInstantiateWithFlags, hipError_t (*)(hipGraphExec_t *, hipGraph_t, unsigned long long));
SAME_FN(hipGraphUpload, hipError_t (*)(hipGraphExec_t, hipStream_t));
SAME_FN(hipGraphLaunch, hipError_t (*)(hipGraphExec_t, hipStream_t));
SAME_FN(hipGraphExecDestroy, hipError_t (*)(hipGraphExec_t));
SAME_FN(hipGraphExecUpdate, hipError_t (*)(hipGraphExec_t, hipGraph_t, hipGraphNode_t *, hipGraphExecUpdateResult *));
SAME_FN(hipGraphExecKernelNodeSetParams, hipError_t (*)(hipGraphExec_t, hipGraphNode_t, const hipKernelNodeParams *));

/* Linked so the object is not dropped; returns the count of checked prototypes. */
int tf_hip_abi_checked(void) { return 63; }
