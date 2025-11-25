#pragma once

#include "onyx/cuda/cuda_runtime_compat.hpp"
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <vector>

namespace onyx::cuda {

// High-throughput host-pinned circular buffer for zero-copy stream ingestion
class PinnedRingBuffer {
public:
    static constexpr size_t DEFAULT_CAPACITY = 64 * 1024 * 1024; // 64 MB

    explicit PinnedRingBuffer(size_t capacity = DEFAULT_CAPACITY);
    ~PinnedRingBuffer();

    PinnedRingBuffer(const PinnedRingBuffer&) = delete;
    PinnedRingBuffer& operator=(const PinnedRingBuffer&) = delete;

    // Push bytes into buffer (producer)
    bool push(const void* data, size_t bytes);

    // Acquire contiguous slice for GPU DMA (consumer)
    struct ReadSlice {
        const void* ptr{nullptr};
        size_t bytes{0};
    };

    ReadSlice peek_available() const;
    void consume(size_t bytes);

    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] bool full() const noexcept { return size() >= capacity_ - 1; }

    void clear();

private:
    uint8_t* buffer_{nullptr};
    size_t capacity_{0};

    // Cache-line separated atomic head and tail pointers
    alignas(CACHE_LINE_BYTES) std::atomic<size_t> head_{0};
    alignas(CACHE_LINE_BYTES) std::atomic<size_t> tail_{0};
};

} // namespace onyx::cuda
