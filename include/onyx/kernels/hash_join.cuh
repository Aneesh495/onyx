#pragma once

#include "onyx/cuda/warp_intrinsics.cuh"
#include "onyx/cuda/cooperative_primitives.cuh"
#include "onyx/common/bit_utils.hpp"
#include <cstdint>

namespace onyx::kernels {

constexpr uint64_t HASH_EMPTY_KEY = 0xFFFFFFFF80000000ULL; // Sentinel for empty bucket
constexpr uint32_t MAX_PROBE_STEPS = 64;

// 64-bit Murmur3-style integer mixer
ONYX_HOST_DEVICE ONYX_INLINE uint64_t hash_murmur64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

struct HashBucket {
    uint64_t key;
    uint32_t row_id;
    uint32_t padding;
};

// Phase 1: Build hash table in device memory
template <typename KeyType>
ONYX_HOST_DEVICE ONYX_INLINE void hash_join_build(
    const KeyType* __restrict__ build_keys,
    size_t num_build_rows,
    HashBucket* __restrict__ table,
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

    for (size_t i = idx; i < num_build_rows; i += stride) {
        uint64_t k = static_cast<uint64_t>(build_keys[i]);
        uint64_t h = hash_murmur64(k);

        for (uint32_t step = 0; step < MAX_PROBE_STEPS; ++step) {
            uint64_t bucket_idx = (h + step * step) & mask;
            HashBucket* b = &table[bucket_idx];

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
            unsigned long long prev = atomicCAS(
                reinterpret_cast<unsigned long long*>(&b->key),
                static_cast<unsigned long long>(HASH_EMPTY_KEY),
                static_cast<unsigned long long>(k));
            if (prev == HASH_EMPTY_KEY || prev == k) {
                b->row_id = static_cast<uint32_t>(i);
                break;
            }
#else
            if (b->key == HASH_EMPTY_KEY || b->key == k) {
                b->key = k;
                b->row_id = static_cast<uint32_t>(i);
                break;
            }
#endif
        }
    }
}

// Phase 2: Probe hash table and emit matched row pairs
template <typename KeyType>
ONYX_HOST_DEVICE ONYX_INLINE void hash_join_probe(
    const KeyType* __restrict__ probe_keys,
    size_t num_probe_rows,
    const HashBucket* __restrict__ table,
    size_t table_capacity,
    uint32_t* __restrict__ out_build_row_ids,
    uint32_t* __restrict__ out_probe_row_ids,
    uint32_t* __restrict__ out_match_count)
{
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
    uint64_t mask = table_capacity - 1;

    size_t idx = bid * bdim + tid;
    size_t stride = bdim * gdim;

    ONYX_SHARED uint32_t shared_block_matches;
    ONYX_SHARED uint32_t shared_block_offset;

    for (size_t row_idx = idx; row_idx < num_probe_rows; row_idx += stride) {
        if (tid == 0) shared_block_matches = 0;

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        __syncthreads();
#endif

        bool matched = false;
        uint32_t matched_build_row = 0;

        uint64_t k = static_cast<uint64_t>(probe_keys[row_idx]);
        uint64_t h = hash_murmur64(k);

        for (uint32_t step = 0; step < MAX_PROBE_STEPS; ++step) {
            uint64_t bucket_idx = (h + step * step) & mask;
            const HashBucket* b = &table[bucket_idx];

            if (b->key == HASH_EMPTY_KEY) {
                break; // Empty bucket, not found
            }
            if (b->key == k) {
                matched = true;
                matched_build_row = b->row_id;
                break;
            }
        }

        uint32_t ballot = cuda::warp_ballot(matched ? 1 : 0);
        uint32_t warp_matches = bit_utils::popcount(ballot);
        uint32_t warp_rank = bit_utils::popcount(ballot & bit_utils::lane_mask_lt(lid));

        ONYX_SHARED uint32_t shared_warp_offsets[32];
        if (lid == 0) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
            shared_warp_offsets[wid] = atomicAdd(&shared_block_matches, warp_matches);
#else
            shared_warp_offsets[wid] = shared_block_matches;
            shared_block_matches += warp_matches;
#endif
        }

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        __syncthreads();
#endif

        if (tid == 0 && shared_block_matches > 0) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
            shared_block_offset = atomicAdd(out_match_count, shared_block_matches);
#else
            shared_block_offset = *out_match_count;
            *out_match_count += shared_block_matches;
#endif
        }

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        __syncthreads();
#endif

        if (matched) {
            uint32_t write_pos = shared_block_offset + shared_warp_offsets[wid] + warp_rank;
            out_build_row_ids[write_pos] = matched_build_row;
            out_probe_row_ids[write_pos] = static_cast<uint32_t>(row_idx);
        }
    }
}

} // namespace onyx::kernels
