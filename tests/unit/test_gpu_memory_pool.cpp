#include "test_framework.hpp"
#include "onyx/cuda/gpu_memory_pool.hpp"
#include <stdexcept>


ONYX_TEST(GpuMemoryPoolAllocDealloc) {
    auto& pool = onyx::cuda::GpuMemoryPool::global();
    auto initial_stats = pool.get_stats();

    void* ptr1 = pool.allocate(1024);
    ONYX_ASSERT(ptr1 != nullptr);

    void* ptr2 = pool.allocate(4096);
    ONYX_ASSERT(ptr2 != nullptr);
    ONYX_ASSERT(ptr1 != ptr2);

    pool.deallocate(ptr1, 1024);
    pool.deallocate(ptr2, 4096);

    auto stats = pool.get_stats();
    ONYX_ASSERT(stats.total_allocations >= initial_stats.total_allocations + 2);
    ONYX_ASSERT(stats.total_deallocations >= initial_stats.total_deallocations + 2);
}

ONYX_TEST(DeviceBufferRAII) {
    {
        onyx::cuda::DeviceBuffer buf(2048);
        ONYX_ASSERT(!buf.empty());
        ONYX_ASSERT_EQ(buf.size(), 2048);
        buf.zero_out();
    }
}
