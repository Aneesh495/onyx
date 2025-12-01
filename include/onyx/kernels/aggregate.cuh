#pragma once

#include "onyx/cuda/warp_intrinsics.cuh"
#include "onyx/cuda/cooperative_primitives.cuh"
#include "onyx/common/types.hpp"
#include <cstdint>
#include <cmath>

namespace onyx::kernels {

enum class AggOp : uint8_t {
    Sum = 0,
    Count,
    Min,
    Max,
    Avg,
    Variance,
    StdDev
};

// Scalar reduction across all rows (no GROUP BY)
template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE void scalar_aggregate_sum(
    const T* __restrict__ data,
    const uint8_t* __restrict__ null_bitmap,
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
        bool valid = (null_bitmap == nullptr) || bit_utils::get_bit(null_bitmap, i);
        if (valid) {
            local_sum += static_cast<double>(data[i]);
            local_count++;
        }
    }

    // Block level reduction in shared memory
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

// Group-by Aggregation with device hash table
struct GroupAggregateSlot {
    uint64_t group_key;
    double sum;
    uint64_t count;
    double min_val;
    double max_val;
};

template <typename KeyType, typename ValType>
ONYX_HOST_DEVICE ONYX_INLINE void grouped_aggregate_hash(
    const KeyType* __restrict__ group_keys,
    const ValType* __restrict__ values,
    const uint8_t* __restrict__ null_bitmap,
    size_t num_rows,
    GroupAggregateSlot* __restrict__ hash_table,
    size_t table_capacity)
{
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
#else
    size_t idx = cuda::g_simt_thread_ctx.block_idx.x * cuda::g_simt_thread_ctx.block_dim.x +
                 cuda::g_simt_thread_ctx.thread_idx.x;
    size_t stride = cuda::g_simt_thread_ctx.block_dim.x * cuda::g_simt_thread_ctx.grid_dim.x;
#endif

    uint64_t mask = table_capacity - 1;

    for (size_t i = idx; i < num_rows; i += stride) {
        bool valid = (null_bitmap == nullptr) || bit_utils::get_bit(null_bitmap, i);
        if (!valid) continue;

        uint64_t k = static_cast<uint64_t>(group_keys[i]);
        double v = static_cast<double>(values[i]);
        uint64_t h = k ^ (k >> 32);

        for (uint32_t step = 0; step < 64; ++step) {
            uint64_t slot_idx = (h + step * step) & mask;
            GroupAggregateSlot* slot = &hash_table[slot_idx];

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
            unsigned long long prev = atomicCAS(
                reinterpret_cast<unsigned long long*>(&slot->group_key),
                static_cast<unsigned long long>(0xFFFFFFFF80000000ULL),
                static_cast<unsigned long long>(k));
            if (prev == 0xFFFFFFFF80000000ULL || prev == k) {
                atomicAdd(&slot->sum, v);
                atomicAdd(reinterpret_cast<unsigned long long*>(&slot->count), 1ULL);
                break;
            }
#else
            if (slot->group_key == 0xFFFFFFFF80000000ULL || slot->group_key == k) {
                if (slot->group_key == 0xFFFFFFFF80000000ULL) {
                    slot->group_key = k;
                    slot->min_val = v;
                    slot->max_val = v;
                }
                slot->sum += v;
                slot->count += 1;
                if (v < slot->min_val) slot->min_val = v;
                if (v > slot->max_val) slot->max_val = v;
                break;
            }
#endif
        }
    }
}

} // namespace onyx::kernels
