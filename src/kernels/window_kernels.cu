#include "onyx/kernels/window.cuh"

namespace onyx::kernels {

template <typename T>
__global__ void window_sliding_average_kernel(
    const T* __restrict__ in_data,
    size_t num_rows,
    uint32_t window_size,
    double* __restrict__ out_avg)
{
    window_sliding_average<T>(in_data, num_rows, window_size, out_avg);
}

template <typename T>
__global__ void window_lag_kernel(
    const T* __restrict__ in_data,
    size_t num_rows,
    uint32_t offset,
    T default_val,
    T* __restrict__ out_data)
{
    window_lag<T>(in_data, num_rows, offset, default_val, out_data);
}

template <typename T>
__global__ void window_lead_kernel(
    const T* __restrict__ in_data,
    size_t num_rows,
    uint32_t offset,
    T default_val,
    T* __restrict__ out_data)
{
    window_lead<T>(in_data, num_rows, offset, default_val, out_data);
}

} // namespace onyx::kernels
