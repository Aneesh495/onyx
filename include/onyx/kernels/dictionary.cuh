#pragma once

#include "onyx/cuda/warp_intrinsics.cuh"
#include "onyx/cuda/cooperative_primitives.cuh"
#include <cstdint>
#include <cstddef>

namespace onyx::kernels {

// Decode dictionary-encoded categorical column using shared memory cache
template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE void dictionary_decode_kernel(
    const uint32_t* __restrict__ encoded_codes,
    const T* __restrict__ dictionary_entries,
    size_t num_dict_entries,
    size_t num_rows,
    T* __restrict__ out_decoded)
{
    // Block shared memory cache for frequently accessed dictionary entries
    ONYX_SHARED T shared_dict[1024];

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t gdim = gridDim.x;
    uint32_t bid = blockIdx.x;
#else
    uint32_t tid = cuda::g_simt_thread_ctx.thread_idx.x;
    uint32_t bdim = cuda::g_simt_thread_ctx.block_dim.x;
    uint32_t gdim = cuda::g_simt_thread_ctx.grid_dim.x;
    uint32_t bid = cuda::g_simt_thread_ctx.block_idx.x;
#endif

    // Cooperatively stage dictionary into shared memory
    size_t stage_limit = (num_dict_entries < 1024) ? num_dict_entries : 1024;
    for (size_t d = tid; d < stage_limit; d += bdim) {
        shared_dict[d] = dictionary_entries[d];
    }

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    __syncthreads();
#endif

    size_t global_stride = static_cast<size_t>(bdim) * gdim;
    size_t global_idx = static_cast<size_t>(bid) * bdim + tid;

    for (size_t i = global_idx; i < num_rows; i += global_stride) {
        uint32_t code = encoded_codes[i];
        if (code < stage_limit) {
            out_decoded[i] = shared_dict[code];
        } else if (code < num_dict_entries) {
            out_decoded[i] = dictionary_entries[code];
        }
    }
}

} // namespace onyx::kernels
