#include "test_framework.hpp"
#include "onyx/execution/query_engine.hpp"
#include <string>
#include <stdexcept>
#include <cmath>


ONYX_TEST(EndToEndFilterAndCount) {
    auto& engine = onyx::execution::QueryEngine::instance();
    std::string csv = "id,latency\n1,50.0\n2,1200.0\n3,800.0\n4,1500.0\n5,2100.0\n";
    auto tbl = onyx::storage::Table::from_csv("network_logs", csv);
    engine.catalog().register_table(tbl);

    auto res = engine.execute_query("SELECT count(id) FROM network_logs WHERE latency > 1000.0;");
    ONYX_ASSERT(res.status.is_ok());
    ONYX_ASSERT(res.table != nullptr);
    ONYX_ASSERT_EQ(res.table->num_rows(), 1);

    double count_val = res.table->batch(0)->column(0)->get_scalar(0).as_double();
    ONYX_ASSERT_EQ(count_val, 3.0);
}

ONYX_TEST(EndToEndFusedSum) {
    auto& engine = onyx::execution::QueryEngine::instance();
    std::string csv = "status,price\n1,10.0\n2,20.0\n1,30.0\n";
    auto tbl = onyx::storage::Table::from_csv("orders", csv);
    engine.catalog().register_table(tbl);

    auto res = engine.execute_query("SELECT sum(price) FROM orders WHERE status = 1;");
    ONYX_ASSERT(res.status.is_ok());
    ONYX_ASSERT(res.table != nullptr);
    double sum_val = res.table->batch(0)->column(0)->get_scalar(0).as_double();
    ONYX_ASSERT(std::abs(sum_val - 40.0) < 1e-6);
}
