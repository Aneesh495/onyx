#pragma once

#include "onyx/cuda/warp_intrinsics.cuh"
#include <cstdint>
#include <cstddef>
#include <cmath>

namespace onyx::kernels {

// Exponential Moving Average (EMA): S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE void exponential_moving_average_kernel(
    const T* __restrict__ in_data,
    size_t num_rows,
    double alpha,
    double* __restrict__ out_ema)
{
    if (num_rows == 0) return;

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Block-level or single thread prefix recursion
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double current = static_cast<double>(in_data[0]);
        out_ema[0] = current;
        for (size_t i = 1; i < num_rows; ++i) {
            current = alpha * static_cast<double>(in_data[i]) + (1.0 - alpha) * current;
            out_ema[i] = current;
        }
    }
#else
    double current = static_cast<double>(in_data[0]);
    out_ema[0] = current;
    for (size_t i = 1; i < num_rows; ++i) {
        current = alpha * static_cast<double>(in_data[i]) + (1.0 - alpha) * current;
        out_ema[i] = current;
    }
#endif
}

} // namespace onyx::kernels
