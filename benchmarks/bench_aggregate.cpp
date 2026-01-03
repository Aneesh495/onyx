#include "onyx/kernels/aggregate.cuh"
#include <vector>
#include <functional>
#include <string>

namespace onyx::kernels {
template <typename T>
void launch_scalar_aggregate(
    const T* data, const uint8_t* null_bitmap, size_t num_rows,
    double* out_sum, uint64_t* out_count, cudaStream_t stream);
}

extern void register_benchmark(std::string name, size_t num_elements, std::function<void()> func);

#define ONYX_BENCH(name, num_elements) \
    static void bench_##name(); \
    static int dummy_##name = (register_benchmark(#name, num_elements, bench_##name), 0); \
    static void bench_##name()

ONYX_BENCH(Aggregate_ScalarReduction, 1000000) {
    static std::vector<double> data(1000000, 1.25);
    double sum = 0.0;
    uint64_t count = 0;

    onyx::kernels::launch_scalar_aggregate<double>(
        data.data(), nullptr, data.size(), &sum, &count, nullptr);
}
