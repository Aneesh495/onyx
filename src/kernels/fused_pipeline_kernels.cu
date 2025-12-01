#include "onyx/kernels/fused_pipeline.cuh"

namespace onyx::kernels {

template <typename KeyType, typename ValType>
__global__ void fused_filter_project_sum_kernel(
    const KeyType* __restrict__ filter_col,
    KeyType filter_literal,
    CompareOp filter_op,
    const ValType* __restrict__ col_a,
    const ValType* __restrict__ col_b,
    size_t num_rows,
    double* __restrict__ out_sum,
    uint64_t* __restrict__ out_count)
{
    fused_filter_project_sum<KeyType, ValType>(
        filter_col, filter_literal, filter_op, col_a, col_b, num_rows, out_sum, out_count);
}

} // namespace onyx::kernels
