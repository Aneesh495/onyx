#include "test_framework.hpp"
#include "onyx/storage/table.hpp"
#include "onyx/storage/catalog.hpp"
#include <stdexcept>


ONYX_TEST(TableFromCsvAndColumnAccess) {
    std::string csv = "id,score\n1,95.5\n2,80.0\n3,88.5\n";
    auto tbl = onyx::storage::Table::from_csv("grades", csv);

    ONYX_ASSERT(tbl != nullptr);
    ONYX_ASSERT_EQ(tbl->num_rows(), 3);
    ONYX_ASSERT_EQ(tbl->schema()->num_fields(), 2);

    auto batch = tbl->batch(0);
    ONYX_ASSERT(batch != nullptr);
    auto col_score = batch->column("score");
    ONYX_ASSERT(col_score != nullptr);
    ONYX_ASSERT_EQ(col_score->get_scalar(0).as_double(), 95.5);
    ONYX_ASSERT_EQ(col_score->get_scalar(1).as_double(), 80.0);
}
