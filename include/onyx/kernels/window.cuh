#pragma once

#include "onyx/cuda/warp_intrinsics.cuh"
#include "onyx/cuda/cooperative_primitives.cuh"
#include <cstdint>

namespace onyx::kernels {

// Cumulative sum (prefix scan)
template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE void window_cumulative_sum(
    const T* __restrict__ in_data,
    size_t num_rows,
    double* __restrict__ out_cumsum)
{
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
    size_t idx = cuda::g_simt_thread_ctx.block_idx.x * cuda::g_simt_thread_ctx.block_dim.x +
                 cuda::g_simt_thread_ctx.thread_idx.x;
#endif

    // Thread 0 computes sequential scan across partition
    if (idx == 0) {
        double run_sum = 0.0;
        for (size_t i = 0; i < num_rows; ++i) {
            run_sum += static_cast<double>(in_data[i]);
            out_cumsum[i] = run_sum;
        }
    }
}

// Sliding window average: rolling window of N preceding rows
template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE void window_sliding_average(
    const T* __restrict__ in_data,
    size_t num_rows,
    uint32_t window_size,
    double* __restrict__ out_avg)
{
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
#else
    size_t idx = cuda::g_simt_thread_ctx.block_idx.x * cuda::g_simt_thread_ctx.block_dim.x +
                 cuda::g_simt_thread_ctx.thread_idx.x;
    size_t stride = cuda::g_simt_thread_ctx.block_dim.x * cuda::g_simt_thread_ctx.grid_dim.x;
#endif

    for (size_t i = idx; i < num_rows; i += stride) {
        size_t start = (i >= window_size) ? (i - window_size + 1) : 0;
        size_t count = i - start + 1;
        double sum = 0.0;
        for (size_t j = start; j <= i; ++j) {
            sum += static_cast<double>(in_data[j]);
        }
        out_avg[i] = sum / static_cast<double>(count);
    }
}

// Lag operator: reads value from `offset` rows prior, or default if out of bounds
template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE void window_lag(
    const T* __restrict__ in_data,
    size_t num_rows,
    uint32_t offset,
    T default_val,
    T* __restrict__ out_data)
{
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
#else
    size_t idx = cuda::g_simt_thread_ctx.block_idx.x * cuda::g_simt_thread_ctx.block_dim.x +
                 cuda::g_simt_thread_ctx.thread_idx.x;
    size_t stride = cuda::g_simt_thread_ctx.block_dim.x * cuda::g_simt_thread_ctx.grid_dim.x;
#endif

    for (size_t i = idx; i < num_rows; i += stride) {
        if (i >= offset) {
            out_data[i] = in_data[i - offset];
        } else {
            out_data[i] = default_val;
        }
    }
}

// Lead operator: reads value from `offset` rows ahead, or default if out of bounds
template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE void window_lead(
    const T* __restrict__ in_data,
    size_t num_rows,
    uint32_t offset,
    T default_val,
    T* __restrict__ out_data)
{
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
#else
    size_t idx = cuda::g_simt_thread_ctx.block_idx.x * cuda::g_simt_thread_ctx.block_dim.x +
                 cuda::g_simt_thread_ctx.thread_idx.x;
    size_t stride = cuda::g_simt_thread_ctx.block_dim.x * cuda::g_simt_thread_ctx.grid_dim.x;
#endif

    for (size_t i = idx; i < num_rows; i += stride) {
        if (i + offset < num_rows) {
            out_data[i] = in_data[i + offset];
        } else {
            out_data[i] = default_val;
        }
    }
}

} // namespace onyx::kernels
