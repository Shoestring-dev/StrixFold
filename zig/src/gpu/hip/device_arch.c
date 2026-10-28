/* The device's exact gfx name from the installed header's hipDeviceProp_t (after ashhart/TensorFold d46d17a
 * zig/src/hip/device_arch.c): the property struct's layout is the header's, never a Zig replica. */
#define _POSIX_C_SOURCE 200809L /* strnlen */
#include <hip/hip_runtime_api.h>
#include <string.h>

typedef hipError_t (*tf_properties_fn)(hipDeviceProp_t *, int);

int tf_hip_device_arch(tf_properties_fn properties, int device, char *out, size_t capacity) {
    hipDeviceProp_t prop;
    memset(&prop, 0, sizeof(prop));
    if (!properties || !out || capacity == 0 || device < 0) return -1;
    hipError_t result = properties(&prop, device);
    if (result != hipSuccess) return (int)result;
    size_t length = strnlen(prop.gcnArchName, sizeof(prop.gcnArchName));
    if (length == 0 || length == sizeof(prop.gcnArchName) || length >= capacity) return -1;
    memcpy(out, prop.gcnArchName, length + 1);
    return 0;
}
