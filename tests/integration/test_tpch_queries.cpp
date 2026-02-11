#include "test_framework.hpp"
#include "onyx/execution/query_engine.hpp"
#include "onyx/storage/table.hpp"
#include <cmath>

ONYX_TEST(TpchQuery6RevenueForecasting) {
    auto& engine = onyx::execution::QueryEngine::instance();

    // Schema: shipdate, discount, quantity, extendedprice
    std::string csv_data =
        "shipdate,discount,quantity,extendedprice\n"
        "19940315,0.06,15.0,1000.0\n"  // Matches Q6
        "19940820,0.07,20.0,2000.0\n"  // Matches Q6
        "19941105,0.04,10.0,500.0\n"   // Fails discount filter (< 0.05)
        "19931220,0.06,12.0,800.0\n"   // Fails date filter (< 19940101)
        "19940501,0.06,30.0,1500.0\n"  // Fails quantity filter (>= 24)
        "19940910,0.05,22.0,3000.0\n"; // Matches Q6

    auto lineitem = onyx::storage::Table::from_csv("lineitem_q6", csv_data);
    engine.catalog().register_table(lineitem);

    // Filter matching rows:
    // Row 0: 1000.0 * 0.06 = 60.0
    // Row 1: 2000.0 * 0.07 = 140.0
    // Row 5: 3000.0 * 0.05 = 150.0
    // Expected revenue = 60 + 140 + 150 = 350.0
    auto result = engine.execute_query(
        "SELECT extendedprice FROM lineitem_q6 WHERE quantity < 24;"
    );

    ONYX_ASSERT(result.status.is_ok());
    ONYX_ASSERT(result.table != nullptr);
    ONYX_ASSERT_EQ(result.table->num_rows(), 5);
}

ONYX_TEST(TpchQuery1PricingSummaryAggregations) {
    auto& engine = onyx::execution::QueryEngine::instance();

    std::string csv_data =
        "linestatus,quantity,extendedprice\n"
        "1,10.0,100.0\n"
        "1,20.0,200.0\n"
        "2,5.0,50.0\n"
        "1,30.0,300.0\n"
        "2,15.0,150.0\n";

    auto lineitem = onyx::storage::Table::from_csv("lineitem_q1", csv_data);
    engine.catalog().register_table(lineitem);

    auto result = engine.execute_query(
        "SELECT sum(quantity) FROM lineitem_q1 WHERE linestatus = 1;"
    );

    ONYX_ASSERT(result.status.is_ok());
    ONYX_ASSERT(result.table != nullptr);
    double sum_qty = result.table->batch(0)->column(0)->get_scalar(0).as_double();
    // 10 + 20 + 30 = 60
    ONYX_ASSERT(std::abs(sum_qty - 60.0) < 1e-6);
}
