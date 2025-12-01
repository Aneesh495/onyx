#pragma once

#include "onyx/cuda/warp_intrinsics.cuh"
#include <cstdint>
#include <cstring>

namespace onyx::kernels {

struct StringOffset {
    uint32_t offset;
    uint32_t length;
};

// Substring matching on GPU: checks if haystack contains needle
ONYX_HOST_DEVICE ONYX_INLINE bool substring_match(
    const char* haystack,
    uint32_t haystack_len,
    const char* needle,
    uint32_t needle_len)
{
    if (needle_len == 0) return true;
    if (haystack_len < needle_len) return false;

    for (uint32_t i = 0; i <= haystack_len - needle_len; ++i) {
        bool match = true;
        for (uint32_t j = 0; j < needle_len; ++j) {
            if (haystack[i + j] != needle[j]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

// Vectorized string substring filter kernel
ONYX_HOST_DEVICE ONYX_INLINE void string_filter_contains(
    const char* __restrict__ char_buffer,
    const StringOffset* __restrict__ string_offsets,
    size_t num_strings,
    const char* __restrict__ pattern,
    uint32_t pattern_len,
    uint32_t* __restrict__ out_row_ids,
    uint32_t* __restrict__ out_count)
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

    uint32_t lid = cuda::lane_id();
    uint32_t wid = cuda::warp_id();

    ONYX_SHARED uint32_t shared_matches;
    ONYX_SHARED uint32_t shared_offset;

    for (size_t i = idx; i < num_strings; i += stride) {
        if (tid == 0) shared_matches = 0;

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        __syncthreads();
#endif

        StringOffset off = string_offsets[i];
        const char* str = char_buffer + off.offset;
        bool is_match = substring_match(str, off.length, pattern, pattern_len);

        uint32_t ballot = cuda::warp_ballot(is_match ? 1 : 0);
        uint32_t warp_matches = bit_utils::popcount(ballot);
        uint32_t warp_rank = bit_utils::popcount(ballot & bit_utils::lane_mask_lt(lid));

        ONYX_SHARED uint32_t shared_warp_offsets[32];
        if (lid == 0) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
            shared_warp_offsets[wid] = atomicAdd(&shared_matches, warp_matches);
#else
            shared_warp_offsets[wid] = shared_matches;
            shared_matches += warp_matches;
#endif
        }

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        __syncthreads();
#endif

        if (tid == 0 && shared_matches > 0) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
            shared_offset = atomicAdd(out_count, shared_matches);
#else
            shared_offset = *out_count;
            *out_count += shared_matches;
#endif
        }

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        __syncthreads();
#endif

        if (is_match) {
            out_row_ids[shared_offset + shared_warp_offsets[wid] + warp_rank] = static_cast<uint32_t>(i);
        }
    }
}

} // namespace onyx::kernels
