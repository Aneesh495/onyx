#include "test_framework.hpp"
#include "onyx/kernels/radix_sort.cuh"
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace onyx::kernels {
template <typename KeyType, typename ValueType>
void launch_radix_sort(KeyType* keys, ValueType* values, size_t num_elements, bool ascending, cudaStream_t stream);
}


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
