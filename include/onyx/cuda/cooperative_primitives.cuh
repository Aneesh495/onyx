#pragma once

#include "onyx/cuda/warp_intrinsics.cuh"
#include <algorithm>

namespace onyx::cuda {

// Warp level tree reductions using warp shuffles
template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE T warp_reduce_sum(T val, uint32_t mask = FULL_WARP_MASK) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += warp_shfl_down(val, offset, mask);
    }
    return val;
}

template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE T warp_reduce_min(T val, uint32_t mask = FULL_WARP_MASK) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = warp_shfl_down(val, offset, mask);
        val = (other < val) ? other : val;
    }
    return val;
}

template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE T warp_reduce_max(T val, uint32_t mask = FULL_WARP_MASK) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = warp_shfl_down(val, offset, mask);
        val = (other > val) ? other : val;
    }
    return val;
}

template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE T warp_reduce_and(T val, uint32_t mask = FULL_WARP_MASK) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val &= warp_shfl_down(val, offset, mask);
    }
    return val;
}

template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE T warp_reduce_or(T val, uint32_t mask = FULL_WARP_MASK) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val |= warp_shfl_down(val, offset, mask);
    }
    return val;
}

// Warp level Kogge-Stone inclusive prefix scan
template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE T warp_scan_inclusive(T val, uint32_t mask = FULL_WARP_MASK) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        T other = warp_shfl_up(val, offset, mask);
        if (lane_id() >= static_cast<uint32_t>(offset)) {
            val += other;
        }
    }
    return val;
}

// Warp level Kogge-Stone exclusive prefix scan
template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE T warp_scan_exclusive(T val, uint32_t mask = FULL_WARP_MASK) {
    T inclusive = warp_scan_inclusive(val, mask);
    T exclusive = warp_shfl_up(inclusive, 1, mask);
    if (lane_id() == 0) {
        exclusive = 0;
    }
    return exclusive;
}

// Thread block level reduction using shared memory for warp leaders
template <typename T, uint32_t BlockSize = 256>
ONYX_HOST_DEVICE ONYX_INLINE T block_reduce_sum(T val, T* shared_warp_sums) {
    constexpr uint32_t NumWarps = BlockSize / 32;
    uint32_t lid = lane_id();
    uint32_t wid = warp_id();

    // Reduce within warp
    val = warp_reduce_sum(val);

    // Warp leader writes to shared memory
    if (lid == 0) {
        shared_warp_sums[wid] = val;
    }

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    __syncthreads();
#endif

    // First warp reads and reduces warp sums
    T block_sum = 0;
    if (wid == 0) {
        T warp_val = (lid < NumWarps) ? shared_warp_sums[lid] : T(0);
        block_sum = warp_reduce_sum(warp_val);
    }

    return block_sum;
}

} // namespace onyx::cuda
