#include "test_framework.hpp"
#include "onyx/kernels/filter.cuh"
#include "onyx/kernels/radix_sort.cuh"
#include "onyx/storage/column.hpp"
#include <vector>
#include <stdexcept>
#include <climits>

namespace onyx::kernels {
template <typename T>
void launch_filter_column(
    const T* col_data, const uint8_t* null_bitmap, size_t num_rows,
    T literal, CompareOp op, uint32_t* out_row_ids, uint32_t* out_count, cudaStream_t stream);

template <typename KeyType, typename ValueType>
void launch_radix_sort(KeyType* keys, ValueType* values, size_t num_elements, bool ascending, cudaStream_t stream);
}


ONYX_TEST(ZeroRowsEdgeCase) {
    uint32_t match_count = 999;
    onyx::kernels::launch_filter_column<int32_t>(
        nullptr, nullptr, 0, 10, onyx::kernels::CompareOp::Equal, nullptr, &match_count, nullptr);
    // Should safely do nothing without crashing
}

ONYX_TEST(SingleElementSortAndFilter) {
    int32_t single_val = 42;
    uint32_t payload = 0;
    onyx::kernels::launch_radix_sort<int32_t, uint32_t>(&single_val, &payload, 1, true, nullptr);
    ONYX_ASSERT_EQ(single_val, 42);

    uint32_t match_id = 0;
    uint32_t count = 0;
    onyx::kernels::launch_filter_column<int32_t>(
        &single_val, nullptr, 1, 42, onyx::kernels::CompareOp::Equal, &match_id, &count, nullptr);
    ONYX_ASSERT_EQ(count, 1);
    ONYX_ASSERT_EQ(match_id, 0);
}

ONYX_TEST(BoundaryValuesMaxInt) {
    std::vector<int32_t> data = {INT32_MAX, 0, INT32_MIN};
    std::vector<uint32_t> payload = {0, 1, 2};

    onyx::kernels::launch_radix_sort<int32_t, uint32_t>(
        data.data(), payload.data(), data.size(), true, nullptr);

    // After sort: INT32_MIN, 0, INT32_MAX
    ONYX_ASSERT_EQ(data[0], INT32_MIN);
    ONYX_ASSERT_EQ(data[1], 0);
    ONYX_ASSERT_EQ(data[2], INT32_MAX);
}
