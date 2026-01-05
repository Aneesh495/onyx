#include "test_framework.hpp"
#include "onyx/kernels/aggregate.cuh"
#include <vector>
#include <stdexcept>
#include <cmath>

namespace onyx::kernels {
template <typename T>
void launch_scalar_aggregate(
    const T* data, const uint8_t* null_bitmap, size_t num_rows,
    double* out_sum, uint64_t* out_count, cudaStream_t stream);
}


ONYX_TEST(ScalarAggregateSum) {
    std::vector<int32_t> data = {10, 20, 30, 40, 50};
    double sum = 0.0;
    uint64_t count = 0;

    onyx::kernels::launch_scalar_aggregate<int32_t>(
        data.data(), nullptr, data.size(), &sum, &count, nullptr);

    ONYX_ASSERT_EQ(count, 5);
    ONYX_ASSERT(std::abs(sum - 150.0) < 1e-6);
}
