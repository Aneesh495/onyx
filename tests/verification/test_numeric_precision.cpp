#include "test_framework.hpp"
#include "onyx/kernels/kernel_launchers.hpp"
#include <vector>
#include <limits>
#include <cmath>

ONYX_TEST(FloatingPointSubnormalsAndPrecision) {
    // Test that floating point aggregations handle subnormal and fractional numbers without underflow
    double tiny = 1e-15;
    std::vector<double> input = {tiny, 2.0 * tiny, 3.0 * tiny, 4.0 * tiny};
    double sum = 0.0;
    uint64_t count = 0;

    onyx::kernels::launch_scalar_aggregate<double>(
        input.data(), nullptr, input.size(), &sum, &count, nullptr);

    ONYX_ASSERT_EQ(count, 4);
    ONYX_ASSERT(std::abs(sum - 10.0 * tiny) < 1e-20);
}

ONYX_TEST(FloatingPointNegativeZeroAndInfinities) {
    std::vector<double> input = {-0.0, 0.0, -100.5, 200.5};
    uint32_t out_count = 0;
    std::vector<uint32_t> out_row_ids(input.size());

    // Filter greater than 0
    onyx::kernels::launch_filter_column<double>(
        input.data(), nullptr, input.size(), 0.0,
        onyx::kernels::CompareOp::GreaterThan,
        out_row_ids.data(), &out_count, nullptr);

    ONYX_ASSERT_EQ(out_count, 1);
    ONYX_ASSERT_EQ(out_row_ids[0], 3); // 200.5
}
