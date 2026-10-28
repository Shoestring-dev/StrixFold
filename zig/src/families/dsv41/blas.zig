//! The fp32 GEMMs the served build runs through torch.matmul (the MoE gate's logits and the indexer's weights of prompt
//! chunks, the DSpark gate and confidence head): cuBLAS on CUDA (cublas.zig), hipBLAS on HIP (hipblas.zig), the same
//! calls and the same row-major-through-column-major arguments.
const gpu = @import("gpu");

pub const Blas = if (gpu.backend == .cuda) @import("cublas.zig").Blas else @import("hipblas.zig").Blas;
