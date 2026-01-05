#include "test_framework.hpp"
#include "onyx/kernels/filter.cuh"
#include <vector>
#include <stdexcept>

namespace onyx::kernels {
template <typename T>
void launch_filter_column(
    const T* col_data, const uint8_t* null_bitmap, size_t num_rows,
    T literal, CompareOp op, uint32_t* out_row_ids, uint32_t* out_count, cudaStream_t stream);
}


ONYX_TEST(FilterKernelGreaterThan) {
    std::vector<int32_t> data = {10, 25, 5, 80, 100, 12, 45, 90};
    std::vector<uint32_t> matched(data.size());
    uint32_t match_count = 0;

    onyx::kernels::launch_filter_column<int32_t>(
        data.data(), nullptr, data.size(), 30,
        onyx::kernels::CompareOp::GreaterThan,
        matched.data(), &match_count, nullptr);

    // 80, 100, 45, 90 match -> 4 rows
    ONYX_ASSERT_EQ(match_count, 4);
}

ONYX_TEST(FilterKernelEqualZeroMatches) {
    std::vector<int32_t> data = {1, 2, 3, 4, 5};
    std::vector<uint32_t> matched(data.size());
    uint32_t match_count = 0;

    onyx::kernels::launch_filter_column<int32_t>(
        data.data(), nullptr, data.size(), 999,
        onyx::kernels::CompareOp::Equal,
        matched.data(), &match_count, nullptr);

    ONYX_ASSERT_EQ(match_count, 0);
}
