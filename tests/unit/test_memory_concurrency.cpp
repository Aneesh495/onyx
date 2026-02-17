#include "test_framework.hpp"
#include "onyx/cuda/gpu_memory_pool.hpp"
#include <thread>
#include <vector>

ONYX_TEST(ConcurrentMemoryPoolAllocationStress) {
    auto& pool = onyx::cuda::GpuMemoryPool::global();
    constexpr int NUM_THREADS = 8;
    constexpr int ALLOCS_PER_THREAD = 250;

    std::vector<std::thread> workers;
    workers.reserve(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; ++t) {
        workers.emplace_back([&pool]() {
            std::vector<std::pair<void*, size_t>> allocations;
            allocations.reserve(ALLOCS_PER_THREAD);

            for (int i = 0; i < ALLOCS_PER_THREAD; ++i) {
                size_t sz = 128 + ((i * 37) % 4096);
                void* ptr = pool.allocate(sz);
                ONYX_ASSERT(ptr != nullptr);
                allocations.push_back({ptr, sz});
            }

            for (const auto& [ptr, sz] : allocations) {
                pool.deallocate(ptr, sz);
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    auto stats = pool.get_stats();
    ONYX_ASSERT(stats.total_allocations >= NUM_THREADS * ALLOCS_PER_THREAD);
    ONYX_ASSERT(stats.total_deallocations >= NUM_THREADS * ALLOCS_PER_THREAD);
}
