#include "onyx/kernels/rolling_window.cuh"
#include "onyx/cuda/simt_engine.hpp"

namespace onyx::kernels {

template <typename T>
void launch_exponential_moving_average(
    const T* in_data,
    size_t num_rows,
    double alpha,
    double* out_ema,
    cudaStream_t stream)
{
    if (num_rows == 0) return;

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA launch
#else
    exponential_moving_average_kernel<T>(in_data, num_rows, alpha, out_ema);
#endif
}

// Instantiations
template void launch_exponential_moving_average<float>(const float*, size_t, double, double*, cudaStream_t);
template void launch_exponential_moving_average<double>(const double*, size_t, double, double*, cudaStream_t);
template void launch_exponential_moving_average<int32_t>(const int32_t*, size_t, double, double*, cudaStream_t);
template void launch_exponential_moving_average<int64_t>(const int64_t*, size_t, double, double*, cudaStream_t);

} // namespace onyx::kernels
