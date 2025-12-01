#include "onyx/kernels/filter.cuh"

namespace onyx::kernels {

template <typename T>
__global__ void filter_column_kernel(
    const T* __restrict__ col_data,
    const uint8_t* __restrict__ null_bitmap,
    size_t num_rows,
    T literal,
    CompareOp op,
    uint32_t* __restrict__ out_row_ids,
    uint32_t* __restrict__ out_count)
{
    filter_column_warp_vote<T>(col_data, null_bitmap, num_rows, literal, op, out_row_ids, out_count);
}

} // namespace onyx::kernels
