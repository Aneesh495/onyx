#pragma once

#include "onyx/cuda/cuda_runtime_compat.hpp"
#include "onyx/common/status.hpp"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <mutex>
#include <atomic>
#include <string>

namespace onyx::cuda {

struct MemoryPoolStats {
    size_t total_reserved_bytes{0};
    size_t active_allocated_bytes{0};
    size_t peak_allocated_bytes{0};
    uint64_t total_allocations{0};
    uint64_t total_deallocations{0};
    size_t num_arenas{0};

    [[nodiscard]] double fragmentation_ratio() const noexcept {
        if (total_reserved_bytes == 0) return 0.0;
        return 1.0 - (static_cast<double>(active_allocated_bytes) / total_reserved_bytes);
    }

    [[nodiscard]] std::string to_string() const;
};

class GpuMemoryPool {
public:
    static constexpr size_t MIN_BLOCK_SIZE = 64;
    static constexpr size_t MAX_BLOCK_SIZE = 64 * 1024 * 1024; // 64 MB
    static constexpr size_t NUM_SIZE_CLASSES = 21; // 64B, 128B, ... 64MB
    static constexpr size_t DEFAULT_ARENA_SIZE = 256 * 1024 * 1024; // 256 MB

    explicit GpuMemoryPool(size_t initial_arena_size = DEFAULT_ARENA_SIZE);
    ~GpuMemoryPool();

    GpuMemoryPool(const GpuMemoryPool&) = delete;
    GpuMemoryPool& operator=(const GpuMemoryPool&) = delete;

    // Allocate memory on device
    [[nodiscard]] void* allocate(size_t bytes);

    // Deallocate memory back to pool
    void deallocate(void* ptr, size_t bytes);

    // Reset all pool arenas
    void reset();

    // Query statistics
    [[nodiscard]] MemoryPoolStats get_stats() const;

    static GpuMemoryPool& global();

private:
    struct FreeBlock {
        FreeBlock* next{nullptr};
    };

    struct Arena {
        void* device_base{nullptr};
        size_t size{0};
        size_t offset{0};
    };

    static size_t size_class_index(size_t bytes) noexcept;
    static size_t block_size_for_class(size_t class_idx) noexcept;

    void* allocate_from_arena(size_t bytes);
    bool create_new_arena(size_t min_bytes);

    mutable std::mutex mutex_;
    size_t default_arena_size_;
    std::vector<Arena> arenas_;
    FreeBlock* free_lists_[NUM_SIZE_CLASSES]{nullptr};

    std::atomic<size_t> active_bytes_{0};
    std::atomic<size_t> peak_bytes_{0};
    std::atomic<uint64_t> alloc_count_{0};
    std::atomic<uint64_t> dealloc_count_{0};
};

// RAII Device Buffer
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes, GpuMemoryPool* pool = &GpuMemoryPool::global());
    ~DeviceBuffer();

    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] void* data() noexcept { return ptr_; }
    [[nodiscard]] const void* data() const noexcept { return ptr_; }
    [[nodiscard]] size_t size() const noexcept { return bytes_; }
    [[nodiscard]] bool empty() const noexcept { return bytes_ == 0 || ptr_ == nullptr; }

    template <typename T>
    [[nodiscard]] T* as() noexcept { return static_cast<T*>(ptr_); }

    template <typename T>
    [[nodiscard]] const T* as() const noexcept { return static_cast<const T*>(ptr_); }

    void resize(size_t new_bytes);
    void zero_out();

private:
    void* ptr_{nullptr};
    size_t bytes_{0};
    GpuMemoryPool* pool_{&GpuMemoryPool::global()};
};

} // namespace onyx::cuda
