#include "test_framework.hpp"
#include "onyx/cuda/gpu_memory_pool.hpp"

ONYX_TEST(DeviceBufferMoveSemanticsAndResize) {
    auto& pool = onyx::cuda::GpuMemoryPool::global();

    // Allocate 1024 bytes buffer
    onyx::cuda::DeviceBuffer buf1(1024, &pool);
    ONYX_ASSERT(!buf1.empty());
    ONYX_ASSERT_EQ(buf1.size(), 1024);
    ONYX_ASSERT(buf1.data() != nullptr);

    // Test Move Constructor
    onyx::cuda::DeviceBuffer buf2(std::move(buf1));
    ONYX_ASSERT(buf1.empty());
    ONYX_ASSERT(buf1.data() == nullptr);
    ONYX_ASSERT_EQ(buf2.size(), 1024);
    ONYX_ASSERT(buf2.data() != nullptr);

    // Test In-place Resize
    buf2.resize(4096);
    ONYX_ASSERT_EQ(buf2.size(), 4096);
    ONYX_ASSERT(buf2.data() != nullptr);

    // Test Zero-byte Allocation Edge Case
    onyx::cuda::DeviceBuffer zero_buf(0, &pool);
    ONYX_ASSERT(zero_buf.empty());
    ONYX_ASSERT(zero_buf.data() == nullptr);
}
