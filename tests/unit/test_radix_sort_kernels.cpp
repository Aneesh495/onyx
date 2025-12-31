#include "onyx/kernels/radix_sort.cuh"
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace onyx::kernels {
template <typename KeyType, typename ValueType>
void launch_radix_sort(KeyType* keys, ValueType* values, size_t num_elements, bool ascending, cudaStream_t stream);
}

extern void register_test(std::string name, std::function<void()> func);

#define ONYX_TEST(name) \
    static void test_##name(); \
    static int dummy_##name = (register_test(#name, test_##name), 0); \
    static void test_##name()

#define ONYX_ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while (0)
#define ONYX_ASSERT_EQ(a, b) do { if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b); } while (0)

ONYX_TEST(RadixSortAscending) {
    std::vector<int32_t> keys = {45, 12, 89, 2, 77, 33, 105, 1};
    std::vector<uint32_t> values = {0, 1, 2, 3, 4, 5, 6, 7};

    onyx::kernels::launch_radix_sort<int32_t, uint32_t>(
        keys.data(), values.data(), keys.size(), true, nullptr);

    ONYX_ASSERT(std::is_sorted(keys.begin(), keys.end()));
    ONYX_ASSERT_EQ(keys[0], 1);
    ONYX_ASSERT_EQ(keys.back(), 105);
}

ONYX_TEST(RadixSortDescending) {
    std::vector<int32_t> keys = {45, 12, 89, 2, 77, 33, 105, 1};
    std::vector<uint32_t> values = {0, 1, 2, 3, 4, 5, 6, 7};

    onyx::kernels::launch_radix_sort<int32_t, uint32_t>(
        keys.data(), values.data(), keys.size(), false, nullptr);

    ONYX_ASSERT_EQ(keys[0], 105);
    ONYX_ASSERT_EQ(keys.back(), 1);
}
