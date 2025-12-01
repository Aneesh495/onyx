#pragma once

#include "onyx/cuda/warp_intrinsics.cuh"
#include "onyx/cuda/cooperative_primitives.cuh"
#include "onyx/common/types.hpp"
#include "onyx/common/bit_utils.hpp"
#include <cstdint>

namespace onyx::kernels {

enum class CompareOp : uint8_t {
    Equal = 0,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    IsNull,
    IsNotNull
};

template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE bool evaluate_predicate(T val, T literal, CompareOp op) {
    switch (op) {
        case CompareOp::Equal:              return val == literal;
        case CompareOp::NotEqual:           return val != literal;
        case CompareOp::LessThan:           return val < literal;
        case CompareOp::LessThanOrEqual:    return val <= literal;
        case CompareOp::GreaterThan:        return val > literal;
        case CompareOp::GreaterThanOrEqual: return val >= literal;
        default:                            return false;
    }
}

// Vectorized SIMT filter kernel with warp voting and prefix compaction
template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE void filter_column_warp_vote(
    const T* __restrict__ col_data,
    const uint8_t* __restrict__ null_bitmap,
    size_t num_rows,
    T literal,
    CompareOp op,
    uint32_t* __restrict__ out_row_ids,
    uint32_t* __restrict__ out_count)
{
    // Block level variables
    ONYX_SHARED uint32_t shared_block_match_count;
    ONYX_SHARED uint32_t shared_block_offset;

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t gdim = gridDim.x;
#else
    uint32_t tid = cuda::g_simt_thread_ctx.thread_idx.x;
    uint32_t bid = cuda::g_simt_thread_ctx.block_idx.x;
    uint32_t bdim = cuda::g_simt_thread_ctx.block_dim.x;
    uint32_t gdim = cuda::g_simt_thread_ctx.grid_dim.x;
#endif

    uint32_t lid = cuda::lane_id();
    uint32_t wid = cuda::warp_id();
    uint32_t total_warps = bdim / 32;

    uint32_t stride = bdim * gdim;
    uint32_t base_row = bid * bdim + tid;

    for (uint32_t row_idx = base_row; row_idx < num_rows; row_idx += stride) {
        if (tid == 0) {
            shared_block_match_count = 0;
        }

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        __syncthreads();
#endif

        bool is_match = false;
        bool is_null = (null_bitmap != nullptr) && !bit_utils::get_bit(null_bitmap, row_idx);

        if (op == CompareOp::IsNull) {
            is_match = is_null;
        } else if (op == CompareOp::IsNotNull) {
            is_match = !is_null;
        } else if (!is_null) {
            is_match = evaluate_predicate(col_data[row_idx], literal, op);
        }

        // Warp ballot: each thread participates in voting
        uint32_t ballot = cuda::warp_ballot(is_match ? 1 : 0);
        uint32_t warp_matches = bit_utils::popcount(ballot);

        // Compute local rank within warp
        uint32_t lane_mask = bit_utils::lane_mask_lt(lid);
        uint32_t warp_rank = bit_utils::popcount(ballot & lane_mask);

        // Warp leader allocates space in block shared counter
        ONYX_SHARED uint32_t shared_warp_offsets[32];
        if (lid == 0) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
            shared_warp_offsets[wid] = atomicAdd(&shared_block_match_count, warp_matches);
#else
            shared_warp_offsets[wid] = shared_block_match_count;
            shared_block_match_count += warp_matches;
#endif
        }

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        __syncthreads();
#endif

        // Thread 0 allocates block matches in global counter
        if (tid == 0 && shared_block_match_count > 0) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
            shared_block_offset = atomicAdd(out_count, shared_block_match_count);
#else
            shared_block_offset = *out_count;
            *out_count += shared_block_match_count;
#endif
        }

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        __syncthreads();
#endif

        // If thread matched, write row index directly to global memory
        if (is_match) {
            uint32_t global_idx = shared_block_offset + shared_warp_offsets[wid] + warp_rank;
            out_row_ids[global_idx] = row_idx;
        }
    }
}

// Disjunctive/conjunctive filter combining two selection bitmasks
ONYX_HOST_DEVICE ONYX_INLINE void combine_selection_masks(
    const uint64_t* __restrict__ mask_a,
    const uint64_t* __restrict__ mask_b,
    size_t num_words,
    bool is_and,
    uint64_t* __restrict__ out_mask)
{
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
#else
    size_t idx = cuda::g_simt_thread_ctx.block_idx.x * cuda::g_simt_thread_ctx.block_dim.x +
                 cuda::g_simt_thread_ctx.thread_idx.x;
    size_t stride = cuda::g_simt_thread_ctx.block_dim.x * cuda::g_simt_thread_ctx.grid_dim.x;
#endif

    for (size_t i = idx; i < num_words; i += stride) {
        out_mask[i] = is_and ? (mask_a[i] & mask_b[i]) : (mask_a[i] | mask_b[i]);
    }
}

} // namespace onyx::kernels
