#include <hip/hip_runtime.h>

// A block normalizes one query row. The score allocation is limited to one tile.
extern "C" __global__ void normalize_scores(float *scores, int keys) {
    __shared__ float reduction[256];
    const int lane = threadIdx.x;
    float *row = scores + static_cast<size_t>(blockIdx.x) * keys;
    float max_value = -INFINITY;
    for (int i = lane; i < keys; i += blockDim.x)
        max_value = fmaxf(max_value, row[i]);
    reduction[lane] = max_value;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step /= 2) {
        if (lane < step)
            reduction[lane] = fmaxf(reduction[lane], reduction[lane + step]);
        __syncthreads();
    }
    max_value = reduction[0];
    __syncthreads();
    float sum = 0;
    for (int i = lane; i < keys; i += blockDim.x) {
        row[i] = expf(row[i] - max_value);
        sum += row[i];
    }
    reduction[lane] = sum;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step /= 2) {
        if (lane < step)
            reduction[lane] += reduction[lane + step];
        __syncthreads();
    }
    const float scale = 1.0f / reduction[0];
    for (int i = lane; i < keys; i += blockDim.x)
        row[i] *= scale;
}
