#include "onyx/kernels/radix_sort.cuh"
#include <vector>
#include <functional>
#include <string>

namespace onyx::kernels {
template <typename KeyType, typename ValueType>
void launch_radix_sort(KeyType* keys, ValueType* values, size_t num_elements, bool ascending, cudaStream_t stream);
}

extern void register_benchmark(std::string name, size_t num_elements, std::function<void()> func);

#define ONYX_BENCH(name, num_elements) \
    static void bench_##name(); \
    static int dummy_##name = (register_benchmark(#name, num_elements, bench_##name), 0); \
    static void bench_##name()

ONYX_BENCH(RadixSort_LSD_KeyVal, 500000) {
    std::vector<int32_t> keys(500000);
    std::vector<uint32_t> payloads(500000);
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = static_cast<int32_t>((i * 1103515245 + 12345) & 0x7FFFFFFF);
        payloads[i] = static_cast<uint32_t>(i);
    }

    onyx::kernels::launch_radix_sort<int32_t, uint32_t>(
        keys.data(), payloads.data(), keys.size(), true, nullptr);
}
