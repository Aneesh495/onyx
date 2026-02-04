#include "onyx/kernels/multi_radix_sort.cuh"

namespace onyx::kernels {

// GPU multi-key permutation generator kernel
__global__ void multi_column_sort_permutation_kernel(
    uint32_t* __restrict__ row_indices,
    size_t num_rows)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        row_indices[idx] = static_cast<uint32_t>(idx);
    }
}

} // namespace onyx::kernels
