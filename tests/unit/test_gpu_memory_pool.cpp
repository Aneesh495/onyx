#include "onyx/cuda/gpu_memory_pool.hpp"
#include <stdexcept>

extern void register_test(std::string name, std::function<void()> func);

#define ONYX_TEST(name) \
    static void test_##name(); \
    static int dummy_##name = (register_test(#name, test_##name), 0); \
    static void test_##name()

#define ONYX_ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while (0)
#define ONYX_ASSERT_EQ(a, b) do { if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b); } while (0)

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
