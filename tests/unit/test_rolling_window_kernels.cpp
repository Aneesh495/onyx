#include "test_framework.hpp"
#include "onyx/kernels/kernel_launchers.hpp"
#include <vector>
#include <cmath>

ONYX_TEST(ExponentialMovingAverageRecurrence) {
    std::vector<double> prices = {100.0, 110.0, 105.0, 115.0};
    double alpha = 0.5;
    std::vector<double> out_ema(prices.size(), 0.0);

    onyx::kernels::launch_exponential_moving_average<double>(
        prices.data(), prices.size(), alpha, out_ema.data(), nullptr);

    // Expected:
    // S0 = 100.0
    // S1 = 0.5 * 110 + 0.5 * 100 = 105.0
    // S2 = 0.5 * 105 + 0.5 * 105 = 105.0
    // S3 = 0.5 * 115 + 0.5 * 105 = 110.0
    std::vector<double> expected = {100.0, 105.0, 105.0, 110.0};

    for (size_t i = 0; i < prices.size(); ++i) {
        ONYX_ASSERT(std::abs(out_ema[i] - expected[i]) < 1e-6);
    }
}
