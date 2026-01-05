#include "test_framework.hpp"
#include "onyx/kernels/window.cuh"
#include <vector>
#include <stdexcept>
#include <cmath>

namespace onyx::kernels {
template <typename T>
void launch_cumulative_sum(const T* in_data, size_t num_rows, double* out_cumsum, cudaStream_t stream);
template <typename T>
void launch_sliding_average(const T* in_data, size_t num_rows, uint32_t window_size, double* out_avg, cudaStream_t stream);
template <typename T>
void launch_lag(const T* in_data, size_t num_rows, uint32_t offset, T default_val, T* out_data, cudaStream_t stream);
template <typename T>
void launch_lead(const T* in_data, size_t num_rows, uint32_t offset, T default_val, T* out_data, cudaStream_t stream);
}


ONYX_TEST(WindowCumulativeSum) {
    std::vector<int32_t> data = {1, 2, 3, 4, 5};
    std::vector<double> cumsum(data.size());

    onyx::kernels::launch_cumulative_sum<int32_t>(data.data(), data.size(), cumsum.data(), nullptr);

    ONYX_ASSERT_EQ(cumsum[0], 1.0);
    ONYX_ASSERT_EQ(cumsum[1], 3.0);
    ONYX_ASSERT_EQ(cumsum[2], 6.0);
    ONYX_ASSERT_EQ(cumsum[3], 10.0);
    ONYX_ASSERT_EQ(cumsum[4], 15.0);
}

ONYX_TEST(WindowLagOperator) {
    std::vector<int32_t> data = {10, 20, 30, 40};
    std::vector<int32_t> out(data.size());

    onyx::kernels::launch_lag<int32_t>(data.data(), data.size(), 1, -1, out.data(), nullptr);

    ONYX_ASSERT_EQ(out[0], -1);
    ONYX_ASSERT_EQ(out[1], 10);
    ONYX_ASSERT_EQ(out[2], 20);
    ONYX_ASSERT_EQ(out[3], 30);
}
