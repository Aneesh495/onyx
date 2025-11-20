#pragma once

#include <cstdint>
#include <bit>

namespace onyx::bit_utils {

// Count set bits (population count)
[[nodiscard]] constexpr inline uint32_t popcount(uint32_t val) noexcept {
    return static_cast<uint32_t>(std::popcount(val));
}

[[nodiscard]] constexpr inline uint32_t popcount64(uint64_t val) noexcept {
    return static_cast<uint32_t>(std::popcount(val));
}

// Count leading zeros
[[nodiscard]] constexpr inline uint32_t countl_zero(uint32_t val) noexcept {
    return static_cast<uint32_t>(std::countl_zero(val));
}

[[nodiscard]] constexpr inline uint32_t countl_zero64(uint64_t val) noexcept {
    return static_cast<uint32_t>(std::countl_zero(val));
}

// Count trailing zeros
[[nodiscard]] constexpr inline uint32_t countr_zero(uint32_t val) noexcept {
    return static_cast<uint32_t>(std::countr_zero(val));
}

[[nodiscard]] constexpr inline uint32_t countr_zero64(uint64_t val) noexcept {
    return static_cast<uint32_t>(std::countr_zero(val));
}

// Round up to nearest power of two
[[nodiscard]] constexpr inline uint64_t next_power_of_two(uint64_t val) noexcept {
    if (val <= 1) return 1;
    return 1ULL << (64 - std::countl_zero(val - 1));
}

// Check if value is a power of two
[[nodiscard]] constexpr inline bool is_power_of_two(uint64_t val) noexcept {
    return val != 0 && (val & (val - 1)) == 0;
}

// Align size or offset up to alignment boundary
[[nodiscard]] constexpr inline size_t align_up(size_t val, size_t alignment) noexcept {
    return (val + alignment - 1) & ~(alignment - 1);
}

// Generate bitmask with set bits from lane 0 up to lane (lane_id - 1)
[[nodiscard]] constexpr inline uint32_t lane_mask_lt(uint32_t lane_id) noexcept {
    if (lane_id >= 32) return 0xFFFFFFFFU;
    return (1U << lane_id) - 1U;
}

// Generate bitmask with set bits from lane 0 up to and including lane_id
[[nodiscard]] constexpr inline uint32_t lane_mask_le(uint32_t lane_id) noexcept {
    if (lane_id >= 31) return 0xFFFFFFFFU;
    return (1U << (lane_id + 1)) - 1U;
}

// Bitmap operations for null indicators
[[nodiscard]] constexpr inline bool get_bit(const uint8_t* bitmap, size_t bit_index) noexcept {
    return (bitmap[bit_index >> 3] & (1U << (bit_index & 7))) != 0;
}

constexpr inline void set_bit(uint8_t* bitmap, size_t bit_index) noexcept {
    bitmap[bit_index >> 3] |= static_cast<uint8_t>(1U << (bit_index & 7));
}

constexpr inline void clear_bit(uint8_t* bitmap, size_t bit_index) noexcept {
    bitmap[bit_index >> 3] &= static_cast<uint8_t>(~(1U << (bit_index & 7)));
}

} // namespace onyx::bit_utils
