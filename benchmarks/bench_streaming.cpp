#include "onyx/cuda/pinned_ring_buffer.hpp"
#include <vector>
#include <functional>
#include <string>

extern void register_benchmark(std::string name, size_t num_elements, std::function<void()> func);

#define ONYX_BENCH(name, num_elements) \
    static void bench_##name(); \
    static int dummy_##name = (register_benchmark(#name, num_elements, bench_##name), 0); \
    static void bench_##name()

ONYX_BENCH(Streaming_RingBuffer_PushConsume, 500000) {
    onyx::cuda::PinnedRingBuffer ring(16 * 1024 * 1024);
    uint64_t record = 0xDEADBEEFCAFEBABEULL;

    for (int i = 0; i < 500000; ++i) {
        ring.push(&record, sizeof(record));
        if (i % 64 == 0) {
            auto slice = ring.peek_available();
            ring.consume(slice.bytes);
        }
    }
}
