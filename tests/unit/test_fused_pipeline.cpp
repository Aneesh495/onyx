#include "onyx/kernels/fused_pipeline.cuh"
#include <vector>
#include <stdexcept>
#include <cmath>

namespace onyx::kernels {
template <typename KeyType, typename ValType>
void launch_fused_filter_project_sum(
    const KeyType* filter_col, KeyType filter_literal, CompareOp filter_op,
    const ValType* col_a, const ValType* col_b, size_t num_rows,
    double* out_sum, uint64_t* out_count, cudaStream_t stream);
}

extern void register_test(std::string name, std::function<void()> func);

#define ONYX_TEST(name) \
    static void test_##name(); \
    static int dummy_##name = (register_test(#name, test_##name), 0); \
    static void test_##name()

#define ONYX_ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while (0)
#define ONYX_ASSERT_EQ(a, b) do { if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b); } while (0)

ONYX_TEST(FusedFilterProjectSum) {
    std::vector<int32_t> region = {1, 2, 1, 3, 1, 2};
    std::vector<double> price = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    std::vector<double> qty = {2.0, 1.0, 3.0, 4.0, 1.0, 2.0};

    double sum = 0.0;
    uint64_t count = 0;

    // Filter region == 1:
    // row 0: 10 * 2 = 20
    // row 2: 30 * 3 = 90
    // row 4: 50 * 1 = 50
    // Total sum = 160, count = 3
    onyx::kernels::launch_fused_filter_project_sum<int32_t, double>(
        region.data(), 1, onyx::kernels::CompareOp::Equal,
        price.data(), qty.data(), region.size(),
        &sum, &count, nullptr);

    ONYX_ASSERT_EQ(count, 3);
    ONYX_ASSERT(std::abs(sum - 160.0) < 1e-6);
}
