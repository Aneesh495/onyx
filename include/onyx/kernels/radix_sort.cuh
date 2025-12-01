#pragma once

#include "onyx/cuda/warp_intrinsics.cuh"
#include "onyx/cuda/cooperative_primitives.cuh"
#include "onyx/common/bit_utils.hpp"
#include <cstdint>

namespace onyx::kernels {

constexpr uint32_t RADIX_BITS = 4;
constexpr uint32_t RADIX_BINS = 1U << RADIX_BITS; // 16 bins
constexpr uint32_t RADIX_MASK = RADIX_BINS - 1;

// Phase 1: Compute local block histograms for the current digit
template <typename KeyType>
ONYX_HOST_DEVICE ONYX_INLINE void radix_sort_histogram(
    const KeyType* __restrict__ keys,
    size_t num_elements,
    uint32_t bit_shift,
    uint32_t* __restrict__ block_histograms,
    uint32_t num_blocks)
{
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t bdim = blockDim.x;
#else
    uint32_t tid = cuda::g_simt_thread_ctx.thread_idx.x;
    uint32_t bid = cuda::g_simt_thread_ctx.block_idx.x;
    uint32_t bdim = cuda::g_simt_thread_ctx.block_dim.x;
#endif

    ONYX_SHARED uint32_t shared_hist[RADIX_BINS];

    if (tid < RADIX_BINS) {
        shared_hist[tid] = 0;
    }

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    __syncthreads();
#endif

    size_t elements_per_block = (num_elements + num_blocks - 1) / num_blocks;
    size_t start_idx = bid * elements_per_block;
    size_t end_idx = std::min(start_idx + elements_per_block, num_elements);

    for (size_t i = start_idx + tid; i < end_idx; i += bdim) {
        uint32_t digit = (static_cast<uint64_t>(keys[i]) >> bit_shift) & RADIX_MASK;
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        atomicAdd(&shared_hist[digit], 1);
#else
        shared_hist[digit]++;
#endif
    }

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    __syncthreads();
#endif

    if (tid < RADIX_BINS) {
        block_histograms[tid * num_blocks + bid] = shared_hist[tid];
    }
}

// Phase 3: Scatter keys and payloads based on global offsets
template <typename KeyType, typename ValueType>
ONYX_HOST_DEVICE ONYX_INLINE void radix_sort_scatter(
    const KeyType* __restrict__ in_keys,
    const ValueType* __restrict__ in_values,
    KeyType* __restrict__ out_keys,
    ValueType* __restrict__ out_values,
    size_t num_elements,
    uint32_t bit_shift,
    const uint32_t* __restrict__ global_offsets,
    uint32_t num_blocks)
{
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t bdim = blockDim.x;
#else
    uint32_t tid = cuda::g_simt_thread_ctx.thread_idx.x;
    uint32_t bid = cuda::g_simt_thread_ctx.block_idx.x;
    uint32_t bdim = cuda::g_simt_thread_ctx.block_dim.x;
#endif

    ONYX_SHARED uint32_t shared_offsets[RADIX_BINS];

    if (tid < RADIX_BINS) {
        shared_offsets[tid] = global_offsets[tid * num_blocks + bid];
    }

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    __syncthreads();
#endif

    size_t elements_per_block = (num_elements + num_blocks - 1) / num_blocks;
    size_t start_idx = bid * elements_per_block;
    size_t end_idx = std::min(start_idx + elements_per_block, num_elements);

    // Thread-serial or warp-cooperative scatter within block
    if (tid == 0) {
        for (size_t i = start_idx; i < end_idx; ++i) {
            KeyType key = in_keys[i];
            ValueType val = in_values ? in_values[i] : ValueType(i);
            uint32_t digit = (static_cast<uint64_t>(key) >> bit_shift) & RADIX_MASK;
            uint32_t dest = shared_offsets[digit]++;
            out_keys[dest] = key;
            if (out_values) {
                out_values[dest] = val;
            }
        }
    }
}

} // namespace onyx::kernels
