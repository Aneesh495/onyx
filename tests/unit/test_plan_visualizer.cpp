#include "test_framework.hpp"
#include "onyx/planner/plan_visualizer.hpp"
#include "onyx/planner/logical_plan.hpp"
#include "onyx/planner/physical_plan.hpp"

ONYX_TEST(PlanVisualizerTreeFormatting) {
    auto scan = std::make_shared<onyx::planner::LogicalScan>("orders", nullptr);
    auto pred = std::make_shared<onyx::sql::BinaryExpr>(
        std::make_shared<onyx::sql::ColumnRefExpr>("status"),
        onyx::sql::BinaryOp::Equal,
        std::make_shared<onyx::sql::LiteralExpr>(onyx::ScalarValue::make_int32(1)));
    auto filter = std::make_shared<onyx::planner::LogicalFilter>(scan, pred);

    std::string text = onyx::planner::PlanVisualizer::format_logical_plan(*filter);
    ONYX_ASSERT(!text.empty());
    ONYX_ASSERT(text.find("LogicalFilter") != std::string::npos);
    ONYX_ASSERT(text.find("LogicalScan") != std::string::npos);
    ONYX_ASSERT(text.find("|--") != std::string::npos);
}
