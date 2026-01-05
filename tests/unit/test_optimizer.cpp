#include "test_framework.hpp"
#include "onyx/planner/optimizer.hpp"
#include "onyx/storage/catalog.hpp"
#include "onyx/sql/parser.hpp"
#include <stdexcept>


ONYX_TEST(OptimizerOperatorFusion) {
    auto& catalog = onyx::storage::Catalog::instance();
    std::string csv = "region_id,price,qty\n1,10.0,2.0\n2,20.0,1.0\n";
    auto table = onyx::storage::Table::from_csv("sales_test", csv);
    catalog.register_table(table);

    std::string sql = "SELECT sum(price * qty) FROM sales_test WHERE region_id = 1;";
    onyx::sql::Parser parser(sql);
    auto stmt = parser.parse_statement();
    auto logical = onyx::planner::LogicalPlanBuilder::build(*stmt, catalog);

    onyx::planner::Optimizer opt;
    auto physical = opt.optimize(logical);

    ONYX_ASSERT(physical != nullptr);
    // Should be fused into PhysicalFusedPipeline
    ONYX_ASSERT_EQ(physical->op_type(), onyx::planner::PhysicalOpType::FusedPipeline);
}
