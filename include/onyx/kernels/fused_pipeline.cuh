#pragma once

#include "onyx/cuda/warp_intrinsics.cuh"
#include "onyx/cuda/cooperative_primitives.cuh"
#include "onyx/kernels/filter.cuh"
#include <cstdint>

namespace onyx::kernels {

// Fused Filter-Project-Sum Aggregate Kernel: evaluates filter, computes a * b, and sums in registers
template <typename KeyType, typename ValType>
ONYX_HOST_DEVICE ONYX_INLINE void fused_filter_project_sum(
    const KeyType* __restrict__ filter_col,
    KeyType filter_literal,
    CompareOp filter_op,
    const ValType* __restrict__ col_a,
    const ValType* __restrict__ col_b,
    size_t num_rows,
    double* __restrict__ out_sum,
    uint64_t* __restrict__ out_count)
{
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    uint32_t tid = threadIdx.x;
#else
    size_t idx = cuda::g_simt_thread_ctx.block_idx.x * cuda::g_simt_thread_ctx.block_dim.x +
                 cuda::g_simt_thread_ctx.thread_idx.x;
    size_t stride = cuda::g_simt_thread_ctx.block_dim.x * cuda::g_simt_thread_ctx.grid_dim.x;
    uint32_t tid = cuda::g_simt_thread_ctx.thread_idx.x;
#endif

    double local_sum = 0.0;
    uint64_t local_count = 0;

    for (size_t i = idx; i < num_rows; i += stride) {
        KeyType key = filter_col[i];
        if (evaluate_predicate(key, filter_literal, filter_op)) {
            double a = static_cast<double>(col_a[i]);
            double b = col_b ? static_cast<double>(col_b[i]) : 1.0;
            local_sum += a * b;
            local_count++;
        }
    }

    ONYX_SHARED double shared_sums[32];
    ONYX_SHARED uint64_t shared_counts[32];

    double block_sum = cuda::block_reduce_sum<double, 256>(local_sum, shared_sums);
    uint64_t block_count = cuda::block_reduce_sum<uint64_t, 256>(local_count, shared_counts);

    if (tid == 0) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        atomicAdd(out_sum, block_sum);
        atomicAdd(reinterpret_cast<unsigned long long*>(out_count), static_cast<unsigned long long>(block_count));
#else
        *out_sum += block_sum;
        *out_count += block_count;
#endif
    }
}

} // namespace onyx::kernels
