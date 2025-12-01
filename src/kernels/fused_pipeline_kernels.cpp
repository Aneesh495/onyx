#include "onyx/kernels/fused_pipeline.cuh"
#include "onyx/cuda/simt_engine.hpp"

namespace onyx::kernels {

template <typename KeyType, typename ValType>
void launch_fused_filter_project_sum(
    const KeyType* filter_col,
    KeyType filter_literal,
    CompareOp filter_op,
    const ValType* col_a,
    const ValType* col_b,
    size_t num_rows,
    double* out_sum,
    uint64_t* out_count,
    cudaStream_t stream)
{
    if (num_rows == 0) {
        *out_sum = 0.0;
        *out_count = 0;
        return;
    }

    uint32_t num_blocks = std::min<uint32_t>((num_rows + 255) / 256, 64);

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA
#else
    cuda::SimtEngine::instance().launch(
        cuda::Dim3(num_blocks), cuda::Dim3(256), 0,
        [&]() {
            fused_filter_project_sum<KeyType, ValType>(
                filter_col, filter_literal, filter_op, col_a, col_b, num_rows, out_sum, out_count);
        });
#endif
}

// Instantiations
template void launch_fused_filter_project_sum<int32_t, double>(const int32_t*, int32_t, CompareOp, const double*, const double*, size_t, double*, uint64_t*, cudaStream_t);
template void launch_fused_filter_project_sum<int64_t, double>(const int64_t*, int64_t, CompareOp, const double*, const double*, size_t, double*, uint64_t*, cudaStream_t);
template void launch_fused_filter_project_sum<int32_t, int32_t>(const int32_t*, int32_t, CompareOp, const int32_t*, const int32_t*, size_t, double*, uint64_t*, cudaStream_t);

} // namespace onyx::kernels
