#include "onyx/kernels/rolling_window.cuh"

namespace onyx::kernels {

template <typename T>
__global__ void exponential_moving_average_cuda_kernel(
    const T* __restrict__ in_data,
    size_t num_rows,
    double alpha,
    double* __restrict__ out_ema)
{
    exponential_moving_average_kernel<T>(in_data, num_rows, alpha, out_ema);
}

} // namespace onyx::kernels
