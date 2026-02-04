#include "test_framework.hpp"
#include "onyx/kernels/kernel_launchers.hpp"
#include <vector>

ONYX_TEST(MultiColumnSortTieBreaking) {
    // Primary key: dept_id (ASC)
    // Secondary key: salary (DESC)
    std::vector<int32_t> dept_id = {10, 20, 10, 20, 10};
    std::vector<double> salary   = {5000.0, 7000.0, 9000.0, 6000.0, 8000.0};
    // Expected order:
    // dept 10: row 2 (9000), row 4 (8000), row 0 (5000)
    // dept 20: row 1 (7000), row 3 (6000)
    // Result permutation: [2, 4, 0, 1, 3]

    std::vector<onyx::kernels::SortKeyDescriptor> keys = {
        {
            .column_data = dept_id.data(),
            .type = onyx::DataType(onyx::TypeId::INT32),
            .direction = onyx::kernels::SortKeyDescriptor::Direction::Ascending,
            .nulls_first = false
        },
        {
            .column_data = salary.data(),
            .type = onyx::DataType(onyx::TypeId::FLOAT64),
            .direction = onyx::kernels::SortKeyDescriptor::Direction::Descending,
            .nulls_first = false
        }
    };

    std::vector<uint32_t> perm(dept_id.size());
    onyx::kernels::launch_multi_column_sort(keys, dept_id.size(), perm.data(), nullptr);

    std::vector<uint32_t> expected = {2, 4, 0, 1, 3};
    for (size_t i = 0; i < perm.size(); ++i) {
        ONYX_ASSERT_EQ(perm[i], expected[i]);
    }
}
