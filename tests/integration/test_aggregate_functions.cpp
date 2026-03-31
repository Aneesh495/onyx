#include "test_framework.hpp"
#include "onyx/execution/query_engine.hpp"
#include "onyx/storage/table.hpp"
#include <cmath>

ONYX_TEST(SqlAggregatesMinMaxCount) {
    auto& engine = onyx::execution::QueryEngine::instance();

    std::string csv = "score\n10.5\n25.0\n5.5\n40.0\n15.0\n";
    auto tbl = onyx::storage::Table::from_csv("scores", csv);
    engine.catalog().register_table(tbl);

    // Count
    auto res_count = engine.execute_query("SELECT count(score) FROM scores;");
    ONYX_ASSERT(res_count.status.is_ok());
    ONYX_ASSERT(res_count.table != nullptr);
    double c = res_count.table->batch(0)->column(0)->get_scalar(0).as_double();
    ONYX_ASSERT_EQ(c, 5.0);

    // Sum
    auto res_sum = engine.execute_query("SELECT sum(score) FROM scores;");
    ONYX_ASSERT(res_sum.status.is_ok());
    double s = res_sum.table->batch(0)->column(0)->get_scalar(0).as_double();
    ONYX_ASSERT(std::abs(s - 96.0) < 1e-6);
}
