#pragma once

#include "onyx/cuda/cuda_runtime_compat.hpp"
#include "onyx/cuda/simt_engine.hpp"
#include "onyx/common/bit_utils.hpp"
#include <cstdint>

namespace onyx::cuda {

constexpr uint32_t FULL_WARP_MASK = 0xFFFFFFFFU;

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1

ONYX_DEVICE ONYX_INLINE uint32_t lane_id() {
    return threadIdx.x & 31;
}

ONYX_DEVICE ONYX_INLINE uint32_t warp_id() {
    return threadIdx.x >> 5;
}

ONYX_DEVICE ONYX_INLINE uint32_t active_mask() {
    return __activemask();
}

ONYX_DEVICE ONYX_INLINE uint32_t warp_ballot(int predicate, uint32_t mask = FULL_WARP_MASK) {
    return __ballot_sync(mask, predicate);
}

ONYX_DEVICE ONYX_INLINE int warp_all(int predicate, uint32_t mask = FULL_WARP_MASK) {
    return __all_sync(mask, predicate);
}

ONYX_DEVICE ONYX_INLINE int warp_any(int predicate, uint32_t mask = FULL_WARP_MASK) {
    return __any_sync(mask, predicate);
}

template <typename T>
ONYX_DEVICE ONYX_INLINE T warp_shfl(T var, int src_lane, uint32_t mask = FULL_WARP_MASK) {
    return __shfl_sync(mask, var, src_lane);
}

template <typename T>
ONYX_DEVICE ONYX_INLINE T warp_shfl_down(T var, unsigned int delta, uint32_t mask = FULL_WARP_MASK) {
    return __shfl_down_sync(mask, var, delta);
}

template <typename T>
ONYX_DEVICE ONYX_INLINE T warp_shfl_up(T var, unsigned int delta, uint32_t mask = FULL_WARP_MASK) {
    return __shfl_up_sync(mask, var, delta);
}

template <typename T>
ONYX_DEVICE ONYX_INLINE T warp_shfl_xor(T var, int lane_mask, uint32_t mask = FULL_WARP_MASK) {
    return __shfl_xor_sync(mask, var, lane_mask);
}

#else // CPU SIMT Emulation

// SIMT Warp State Context for host thread
struct SimtWarpContext {
    uint32_t lane_idx{0};
    uint32_t warp_idx{0};
    uint32_t current_mask{FULL_WARP_MASK};
    uint64_t registers[32]{0};
};

extern thread_local SimtWarpContext g_simt_warp_ctx;

ONYX_HOST_DEVICE ONYX_INLINE uint32_t lane_id() {
    return g_simt_warp_ctx.lane_idx;
}

ONYX_HOST_DEVICE ONYX_INLINE uint32_t warp_id() {
    return g_simt_warp_ctx.warp_idx;
}

ONYX_HOST_DEVICE ONYX_INLINE uint32_t active_mask() {
    return g_simt_warp_ctx.current_mask;
}

ONYX_HOST_DEVICE ONYX_INLINE uint32_t warp_ballot(int predicate, uint32_t mask = FULL_WARP_MASK) {
    return predicate ? (mask & FULL_WARP_MASK) : 0U;
}

ONYX_HOST_DEVICE ONYX_INLINE int warp_all(int predicate, uint32_t mask = FULL_WARP_MASK) {
    uint32_t b = warp_ballot(predicate, mask);
    return (b & mask) == mask;
}

ONYX_HOST_DEVICE ONYX_INLINE int warp_any(int predicate, uint32_t mask = FULL_WARP_MASK) {
    uint32_t b = warp_ballot(predicate, mask);
    return (b & mask) != 0;
}

template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE T warp_shfl(T var, int src_lane, [[maybe_unused]] uint32_t mask = FULL_WARP_MASK) {
    static_assert(sizeof(T) <= sizeof(uint64_t));
    uint64_t raw = 0;
    std::memcpy(&raw, &var, sizeof(T));
    g_simt_warp_ctx.registers[g_simt_warp_ctx.lane_idx] = raw;
    
    uint64_t src_raw = g_simt_warp_ctx.registers[src_lane & 31];
    T res{};
    std::memcpy(&res, &src_raw, sizeof(T));
    return res;
}

template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE T warp_shfl_down(T var, unsigned int delta, [[maybe_unused]] uint32_t mask = FULL_WARP_MASK) {
    static_assert(sizeof(T) <= sizeof(uint64_t));
    uint64_t raw = 0;
    std::memcpy(&raw, &var, sizeof(T));
    g_simt_warp_ctx.registers[g_simt_warp_ctx.lane_idx] = raw;
    
    uint32_t target_lane = g_simt_warp_ctx.lane_idx + delta;
    uint64_t src_raw = (target_lane < 32) ? g_simt_warp_ctx.registers[target_lane] : raw;
    T res{};
    std::memcpy(&res, &src_raw, sizeof(T));
    return res;
}

template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE T warp_shfl_up(T var, unsigned int delta, [[maybe_unused]] uint32_t mask = FULL_WARP_MASK) {
    static_assert(sizeof(T) <= sizeof(uint64_t));
    uint64_t raw = 0;
    std::memcpy(&raw, &var, sizeof(T));
    g_simt_warp_ctx.registers[g_simt_warp_ctx.lane_idx] = raw;
    
    int target_lane = static_cast<int>(g_simt_warp_ctx.lane_idx) - static_cast<int>(delta);
    uint64_t src_raw = (target_lane >= 0) ? g_simt_warp_ctx.registers[target_lane] : raw;
    T res{};
    std::memcpy(&res, &src_raw, sizeof(T));
    return res;
}

template <typename T>
ONYX_HOST_DEVICE ONYX_INLINE T warp_shfl_xor(T var, int lane_mask, [[maybe_unused]] uint32_t mask = FULL_WARP_MASK) {
    static_assert(sizeof(T) <= sizeof(uint64_t));
    uint64_t raw = 0;
    std::memcpy(&raw, &var, sizeof(T));
    g_simt_warp_ctx.registers[g_simt_warp_ctx.lane_idx] = raw;
    
    uint32_t target_lane = g_simt_warp_ctx.lane_idx ^ lane_mask;
    uint64_t src_raw = (target_lane < 32) ? g_simt_warp_ctx.registers[target_lane] : raw;
    T res{};
    std::memcpy(&res, &src_raw, sizeof(T));
    return res;
}

#endif // ONYX_HAS_CUDA

} // namespace onyx::cuda
