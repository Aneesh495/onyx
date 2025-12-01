#pragma once

#include "onyx/cuda/warp_intrinsics.cuh"
#include <cstdint>

namespace onyx::kernels {

// Bit-packing unpacking kernel: unpacks integers of width `bit_width` (1..16) from 64-bit words
ONYX_HOST_DEVICE ONYX_INLINE void bitpack_unpack_u32(
    const uint64_t* __restrict__ packed_words,
    uint32_t bit_width,
    size_t num_elements,
    uint32_t* __restrict__ out_unpacked)
{
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
#else
    size_t idx = cuda::g_simt_thread_ctx.block_idx.x * cuda::g_simt_thread_ctx.block_dim.x +
                 cuda::g_simt_thread_ctx.thread_idx.x;
    size_t stride = cuda::g_simt_thread_ctx.block_dim.x * cuda::g_simt_thread_ctx.grid_dim.x;
#endif

    uint32_t mask = (1U << bit_width) - 1U;

    for (size_t i = idx; i < num_elements; i += stride) {
        size_t bit_offset = i * bit_width;
        size_t word_idx = bit_offset / 64;
        uint32_t bit_in_word = static_cast<uint32_t>(bit_offset % 64);

        uint64_t w0 = packed_words[word_idx];
        uint32_t val = static_cast<uint32_t>((w0 >> bit_in_word) & mask);

        // Check if value spans two 64-bit words
        if (bit_in_word + bit_width > 64) {
            uint64_t w1 = packed_words[word_idx + 1];
            uint32_t bits_from_w0 = 64 - bit_in_word;
            uint32_t bits_from_w1 = bit_width - bits_from_w0;
            uint32_t part1 = static_cast<uint32_t>((w1 & ((1U << bits_from_w1) - 1U)) << bits_from_w0);
            val |= part1;
        }

        out_unpacked[i] = val;
    }
}

// Frame of Reference (FoR) decoding: out = base + delta
ONYX_HOST_DEVICE ONYX_INLINE void frame_of_reference_decode(
    int64_t base_value,
    const int32_t* __restrict__ deltas,
    size_t num_elements,
    int64_t* __restrict__ out_values)
{
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
#else
    size_t idx = cuda::g_simt_thread_ctx.block_idx.x * cuda::g_simt_thread_ctx.block_dim.x +
                 cuda::g_simt_thread_ctx.thread_idx.x;
    size_t stride = cuda::g_simt_thread_ctx.block_dim.x * cuda::g_simt_thread_ctx.grid_dim.x;
#endif

    for (size_t i = idx; i < num_elements; i += stride) {
        out_values[i] = base_value + static_cast<int64_t>(deltas[i]);
    }
}

// Run-Length Encoding (RLE) decoding using prefix offsets and binary search
template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE void rle_decode(
    const T* __restrict__ run_values,
    const uint32_t* __restrict__ run_cumulative_lengths,
    size_t num_runs,
    size_t total_elements,
    T* __restrict__ out_uncompressed)
{
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
#else
    size_t idx = cuda::g_simt_thread_ctx.block_idx.x * cuda::g_simt_thread_ctx.block_dim.x +
                 cuda::g_simt_thread_ctx.thread_idx.x;
    size_t stride = cuda::g_simt_thread_ctx.block_dim.x * cuda::g_simt_thread_ctx.grid_dim.x;
#endif

    for (size_t i = idx; i < total_elements; i += stride) {
        // Binary search for run containing element i
        size_t low = 0;
        size_t high = num_runs;
        while (low < high) {
            size_t mid = low + (high - low) / 2;
            if (run_cumulative_lengths[mid] <= i) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        out_uncompressed[i] = run_values[low];
    }
}

} // namespace onyx::kernels
