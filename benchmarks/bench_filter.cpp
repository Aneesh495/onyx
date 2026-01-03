#include "onyx/kernels/filter.cuh"
#include <vector>
#include <functional>
#include <string>

namespace onyx::kernels {
template <typename T>
void launch_filter_column(
    const T* col_data, const uint8_t* null_bitmap, size_t num_rows,
    T literal, CompareOp op, uint32_t* out_row_ids, uint32_t* out_count, cudaStream_t stream);
}

extern void register_benchmark(std::string name, size_t num_elements, std::function<void()> func);

#define ONYX_BENCH(name, num_elements) \
    static void bench_##name(); \
    static int dummy_##name = (register_benchmark(#name, num_elements, bench_##name), 0); \
    static void bench_##name()

ONYX_BENCH(Filter_Numeric_WarpVote, 1000000) {
    static std::vector<int32_t> data(1000000, 42);
    static std::vector<uint32_t> matches(1000000);
    uint32_t match_count = 0;

    onyx::kernels::launch_filter_column<int32_t>(
        data.data(), nullptr, data.size(), 40,
        onyx::kernels::CompareOp::GreaterThan,
        matches.data(), &match_count, nullptr);
}
