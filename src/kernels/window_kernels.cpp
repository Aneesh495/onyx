#include "onyx/kernels/window.cuh"
#include "onyx/cuda/simt_engine.hpp"

namespace onyx::kernels {

template <typename T>
void launch_cumulative_sum(
    const T* in_data,
    size_t num_rows,
    double* out_cumsum,
    cudaStream_t stream)
{
    if (num_rows == 0) return;

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA
#else
    cuda::SimtEngine::instance().launch(
        cuda::Dim3(1), cuda::Dim3(1), 0,
        [&]() {
            window_cumulative_sum<T>(in_data, num_rows, out_cumsum);
        });
#endif
}

template <typename T>
void launch_sliding_average(
    const T* in_data,
    size_t num_rows,
    uint32_t window_size,
    double* out_avg,
    cudaStream_t stream)
{
    if (num_rows == 0) return;
    uint32_t num_blocks = std::min<uint32_t>((num_rows + 255) / 256, 64);

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA
#else
    cuda::SimtEngine::instance().launch(
        cuda::Dim3(num_blocks), cuda::Dim3(256), 0,
        [&]() {
            window_sliding_average<T>(in_data, num_rows, window_size, out_avg);
        });
#endif
}

template <typename T>
void launch_lag(
    const T* in_data,
    size_t num_rows,
    uint32_t offset,
    T default_val,
    T* out_data,
    cudaStream_t stream)
{
    if (num_rows == 0) return;
    uint32_t num_blocks = std::min<uint32_t>((num_rows + 255) / 256, 64);

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA
#else
    cuda::SimtEngine::instance().launch(
        cuda::Dim3(num_blocks), cuda::Dim3(256), 0,
        [&]() {
            window_lag<T>(in_data, num_rows, offset, default_val, out_data);
        });
#endif
}

template <typename T>
void launch_lead(
    const T* in_data,
    size_t num_rows,
    uint32_t offset,
    T default_val,
    T* out_data,
    cudaStream_t stream)
{
    if (num_rows == 0) return;
    uint32_t num_blocks = std::min<uint32_t>((num_rows + 255) / 256, 64);

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA
#else
    cuda::SimtEngine::instance().launch(
        cuda::Dim3(num_blocks), cuda::Dim3(256), 0,
        [&]() {
            window_lead<T>(in_data, num_rows, offset, default_val, out_data);
        });
#endif
}

// Instantiations
template void launch_cumulative_sum<int32_t>(const int32_t*, size_t, double*, cudaStream_t);
template void launch_cumulative_sum<int64_t>(const int64_t*, size_t, double*, cudaStream_t);
template void launch_cumulative_sum<double>(const double*, size_t, double*, cudaStream_t);

template void launch_sliding_average<int32_t>(const int32_t*, size_t, uint32_t, double*, cudaStream_t);
template void launch_sliding_average<int64_t>(const int64_t*, size_t, uint32_t, double*, cudaStream_t);
template void launch_sliding_average<double>(const double*, size_t, uint32_t, double*, cudaStream_t);

template void launch_lag<int32_t>(const int32_t*, size_t, uint32_t, int32_t, int32_t*, cudaStream_t);
template void launch_lag<int64_t>(const int64_t*, size_t, uint32_t, int64_t, int64_t*, cudaStream_t);
template void launch_lag<double>(const double*, size_t, uint32_t, double, double*, cudaStream_t);

template void launch_lead<int32_t>(const int32_t*, size_t, uint32_t, int32_t, int32_t*, cudaStream_t);
template void launch_lead<int64_t>(const int64_t*, size_t, uint32_t, int64_t, int64_t*, cudaStream_t);
template void launch_lead<double>(const double*, size_t, uint32_t, double, double*, cudaStream_t);

} // namespace onyx::kernels
