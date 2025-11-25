#include "onyx/cuda/pinned_ring_buffer.hpp"
#include "onyx/common/bit_utils.hpp"
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace onyx::cuda {

PinnedRingBuffer::PinnedRingBuffer(size_t capacity)
    : capacity_(bit_utils::next_power_of_two(std::max(capacity, static_cast<size_t>(1024)))) {
    // Allocate 128-byte aligned pinned memory
    void* ptr = nullptr;
    int res = posix_memalign(&ptr, 128, capacity_);
    if (res == 0 && ptr) {
        buffer_ = static_cast<uint8_t*>(ptr);
    }
}

PinnedRingBuffer::~PinnedRingBuffer() {
    if (buffer_) {
        free(buffer_);
        buffer_ = nullptr;
    }
}

size_t PinnedRingBuffer::size() const noexcept {
    size_t h = head_.load(std::memory_order_acquire);
    size_t t = tail_.load(std::memory_order_acquire);
    if (h >= t) {
        return h - t;
    }
    return capacity_ - (t - h);
}

bool PinnedRingBuffer::push(const void* data, size_t bytes) {
    if (!data || bytes == 0 || !buffer_) return false;

    size_t current_size = size();
    if (current_size + bytes >= capacity_) {
        return false; // Not enough space
    }

    size_t h = head_.load(std::memory_order_relaxed);
    const uint8_t* src = static_cast<const uint8_t*>(data);

    // Check if write wraps around end of ring
    size_t space_to_end = capacity_ - h;
    if (bytes <= space_to_end) {
        std::memcpy(buffer_ + h, src, bytes);
        head_.store((h + bytes) & (capacity_ - 1), std::memory_order_release);
    } else {
        std::memcpy(buffer_ + h, src, space_to_end);
        std::memcpy(buffer_, src + space_to_end, bytes - space_to_end);
        head_.store(bytes - space_to_end, std::memory_order_release);
    }

    return true;
}

PinnedRingBuffer::ReadSlice PinnedRingBuffer::peek_available() const {
    if (empty() || !buffer_) return {nullptr, 0};

    size_t h = head_.load(std::memory_order_acquire);
    size_t t = tail_.load(std::memory_order_relaxed);

    size_t contiguous_bytes = 0;
    if (h >= t) {
        contiguous_bytes = h - t;
    } else {
        contiguous_bytes = capacity_ - t;
    }

    return {buffer_ + t, contiguous_bytes};
}

void PinnedRingBuffer::consume(size_t bytes) {
    if (bytes == 0) return;
    size_t t = tail_.load(std::memory_order_relaxed);
    tail_.store((t + bytes) & (capacity_ - 1), std::memory_order_release);
}

void PinnedRingBuffer::clear() {
    head_.store(0, std::memory_order_release);
    tail_.store(0, std::memory_order_release);
}

} // namespace onyx::cuda
