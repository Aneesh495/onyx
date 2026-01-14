#include "onyx/kernels/filter.cuh"
#include "onyx/cuda/simt_engine.hpp"

namespace onyx::kernels {

// Explicit host dispatchers for filter operations across data types
template <typename T>
void launch_filter_column(
    const T* col_data,
    const uint8_t* null_bitmap,
    size_t num_rows,
    T literal,
    CompareOp op,
    uint32_t* out_row_ids,
    uint32_t* out_count,
    cudaStream_t stream)
{
    if (num_rows == 0) return;

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    uint32_t block_size = 256;
    uint32_t grid_size = std::min<uint32_t>((num_rows + block_size - 1) / block_size, 1024);
    // CUDA kernel launch
#else
    uint32_t count = 0;
    for (size_t i = 0; i < num_rows; ++i) {
        bool is_null = (null_bitmap != nullptr) && !bit_utils::get_bit(null_bitmap, i);
        bool is_match = false;
        if (op == CompareOp::IsNull) {
            is_match = is_null;
        } else if (op == CompareOp::IsNotNull) {
            is_match = !is_null;
        } else if (!is_null) {
            is_match = evaluate_predicate(col_data[i], literal, op);
        }
        if (is_match) {
            out_row_ids[count++] = static_cast<uint32_t>(i);
        }
    }
    *out_count = count;
#endif
}

// Instantiations
template void launch_filter_column<int8_t>(const int8_t*, const uint8_t*, size_t, int8_t, CompareOp, uint32_t*, uint32_t*, cudaStream_t);
template void launch_filter_column<int16_t>(const int16_t*, const uint8_t*, size_t, int16_t, CompareOp, uint32_t*, uint32_t*, cudaStream_t);
template void launch_filter_column<int32_t>(const int32_t*, const uint8_t*, size_t, int32_t, CompareOp, uint32_t*, uint32_t*, cudaStream_t);
template void launch_filter_column<int64_t>(const int64_t*, const uint8_t*, size_t, int64_t, CompareOp, uint32_t*, uint32_t*, cudaStream_t);
template void launch_filter_column<float>(const float*, const uint8_t*, size_t, float, CompareOp, uint32_t*, uint32_t*, cudaStream_t);
template void launch_filter_column<double>(const double*, const uint8_t*, size_t, double, CompareOp, uint32_t*, uint32_t*, cudaStream_t);

} // namespace onyx::kernels
