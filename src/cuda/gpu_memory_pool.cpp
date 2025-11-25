#include "onyx/cuda/gpu_memory_pool.hpp"
#include "onyx/common/bit_utils.hpp"
#include "onyx/common/logger.hpp"
#include <sstream>
#include <iomanip>

namespace onyx::cuda {

std::string MemoryPoolStats::to_string() const {
    std::ostringstream oss;
    oss << "MemoryPoolStats { reserved: " << (total_reserved_bytes / (1024 * 1024)) << " MB"
        << ", active: " << (active_allocated_bytes / (1024 * 1024)) << " MB"
        << ", peak: " << (peak_allocated_bytes / (1024 * 1024)) << " MB"
        << ", allocations: " << total_allocations
        << ", deallocations: " << total_deallocations
        << ", arenas: " << num_arenas
        << ", fragmentation: " << std::fixed << std::setprecision(2) << (fragmentation_ratio() * 100.0) << "% }";
    return oss.str();
}

GpuMemoryPool::GpuMemoryPool(size_t initial_arena_size)
    : default_arena_size_(std::max(initial_arena_size, DEFAULT_ARENA_SIZE)) {
    create_new_arena(default_arena_size_);
}

GpuMemoryPool::~GpuMemoryPool() {
    reset();
}

GpuMemoryPool& GpuMemoryPool::global() {
    static GpuMemoryPool pool;
    return pool;
}

size_t GpuMemoryPool::size_class_index(size_t bytes) noexcept {
    if (bytes <= MIN_BLOCK_SIZE) return 0;
    size_t rounded = bit_utils::next_power_of_two(bytes);
    size_t idx = bit_utils::countr_zero64(rounded) - bit_utils::countr_zero64(MIN_BLOCK_SIZE);
    return std::min(idx, NUM_SIZE_CLASSES - 1);
}

size_t GpuMemoryPool::block_size_for_class(size_t class_idx) noexcept {
    return MIN_BLOCK_SIZE << class_idx;
}

bool GpuMemoryPool::create_new_arena(size_t min_bytes) {
    size_t alloc_bytes = std::max(min_bytes, default_arena_size_);
    void* dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, alloc_bytes);
    if (err != cudaSuccess || !dev_ptr) {
        ONYX_LOG_ERROR("GpuMemoryPool failed to allocate arena of size: " << alloc_bytes);
        return false;
    }

    arenas_.push_back(Arena{
        .device_base = dev_ptr,
        .size = alloc_bytes,
        .offset = 0
    });
    return true;
}

void* GpuMemoryPool::allocate_from_arena(size_t bytes) {
    size_t aligned_bytes = bit_utils::align_up(bytes, 128);

    for (auto& arena : arenas_) {
        if (arena.size - arena.offset >= aligned_bytes) {
            void* res = static_cast<char*>(arena.device_base) + arena.offset;
            arena.offset += aligned_bytes;
            return res;
        }
    }

    if (create_new_arena(aligned_bytes)) {
        auto& arena = arenas_.back();
        void* res = arena.device_base;
        arena.offset = aligned_bytes;
        return res;
    }

    return nullptr;
}

void* GpuMemoryPool::allocate(size_t bytes) {
    if (bytes == 0) return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);
    size_t class_idx = size_class_index(bytes);
    size_t class_bytes = block_size_for_class(class_idx);

    if (free_lists_[class_idx] != nullptr) {
        FreeBlock* block = free_lists_[class_idx];
        free_lists_[class_idx] = block->next;

        active_bytes_.fetch_add(class_bytes, std::memory_order_relaxed);
        alloc_count_.fetch_add(1, std::memory_order_relaxed);

        size_t cur = active_bytes_.load(std::memory_order_relaxed);
        size_t prev_peak = peak_bytes_.load(std::memory_order_relaxed);
        while (cur > prev_peak && !peak_bytes_.compare_exchange_weak(prev_peak, cur)) {}

        return block;
    }

    void* ptr = allocate_from_arena(class_bytes);
    if (ptr) {
        active_bytes_.fetch_add(class_bytes, std::memory_order_relaxed);
        alloc_count_.fetch_add(1, std::memory_order_relaxed);

        size_t cur = active_bytes_.load(std::memory_order_relaxed);
        size_t prev_peak = peak_bytes_.load(std::memory_order_relaxed);
        while (cur > prev_peak && !peak_bytes_.compare_exchange_weak(prev_peak, cur)) {}
    }
    return ptr;
}

void GpuMemoryPool::deallocate(void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) return;

    std::lock_guard<std::mutex> lock(mutex_);
    size_t class_idx = size_class_index(bytes);
    size_t class_bytes = block_size_for_class(class_idx);

    FreeBlock* block = static_cast<FreeBlock*>(ptr);
    block->next = free_lists_[class_idx];
    free_lists_[class_idx] = block;

    active_bytes_.fetch_sub(class_bytes, std::memory_order_relaxed);
    dealloc_count_.fetch_add(1, std::memory_order_relaxed);
}

void GpuMemoryPool::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        free_lists_[i] = nullptr;
    }
    for (auto& arena : arenas_) {
        if (arena.device_base) {
            cudaFree(arena.device_base);
        }
    }
    arenas_.clear();
    active_bytes_.store(0, std::memory_order_relaxed);
}

MemoryPoolStats GpuMemoryPool::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total_reserved = 0;
    for (const auto& arena : arenas_) {
        total_reserved += arena.size;
    }

    return MemoryPoolStats{
        .total_reserved_bytes = total_reserved,
        .active_allocated_bytes = active_bytes_.load(std::memory_order_relaxed),
        .peak_allocated_bytes = peak_bytes_.load(std::memory_order_relaxed),
        .total_allocations = alloc_count_.load(std::memory_order_relaxed),
        .total_deallocations = dealloc_count_.load(std::memory_order_relaxed),
        .num_arenas = arenas_.size()
    };
}

DeviceBuffer::DeviceBuffer(size_t bytes, GpuMemoryPool* pool)
    : bytes_(bytes), pool_(pool) {
    if (bytes_ > 0 && pool_) {
        ptr_ = pool_->allocate(bytes_);
    }
}

DeviceBuffer::~DeviceBuffer() {
    if (ptr_ && pool_ && bytes_ > 0) {
        pool_->deallocate(ptr_, bytes_);
        ptr_ = nullptr;
        bytes_ = 0;
    }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_), pool_(other.pool_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        if (ptr_ && pool_ && bytes_ > 0) {
            pool_->deallocate(ptr_, bytes_);
        }
        ptr_ = other.ptr_;
        bytes_ = other.bytes_;
        pool_ = other.pool_;
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

void DeviceBuffer::resize(size_t new_bytes) {
    if (new_bytes == bytes_) return;
    if (ptr_ && pool_ && bytes_ > 0) {
        pool_->deallocate(ptr_, bytes_);
        ptr_ = nullptr;
        bytes_ = 0;
    }
    bytes_ = new_bytes;
    if (bytes_ > 0 && pool_) {
        ptr_ = pool_->allocate(bytes_);
    }
}

void DeviceBuffer::zero_out() {
    if (ptr_ && bytes_ > 0) {
        cudaMemset(ptr_, 0, bytes_);
    }
}

} // namespace onyx::cuda
