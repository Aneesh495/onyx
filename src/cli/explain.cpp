#include "onyx/cli/explain.hpp"
#include "onyx/sql/parser.hpp"
#include "onyx/planner/logical_plan.hpp"
#include "onyx/planner/optimizer.hpp"
#include "onyx/planner/plan_visualizer.hpp"
#include "onyx/storage/catalog.hpp"
#include <sstream>

namespace onyx::cli {

std::string QueryExplainer::explain_query(const std::string& sql, bool verbose) {
    std::ostringstream ss;
    ss << "=== Query Execution Explanation ===\n";
    ss << "SQL: " << sql << "\n\n";

    try {
        sql::Parser parser(sql);
        auto stmt = parser.parse_statement();
        if (!stmt) {
            ss << "Error: Failed to parse SQL statement.\n";
            return ss.str();
        }

        auto& catalog = storage::Catalog::instance();
        auto logical_plan = planner::LogicalPlanBuilder::build(*stmt, catalog);
        if (!logical_plan) {
            ss << "Error: Failed to construct logical plan.\n";
            return ss.str();
        }

        ss << "[Logical Plan Tree]\n";
        ss << planner::PlanVisualizer::format_logical_plan(*logical_plan);
        ss << "\n";

        planner::Optimizer optimizer;
        auto physical_plan = optimizer.optimize(logical_plan);
        if (!physical_plan) {
            ss << "Error: Failed to compile physical plan.\n";
            return ss.str();
        }

        ss << "[Optimized Physical Execution DAG]\n";
        ss << planner::PlanVisualizer::format_physical_plan(*physical_plan);

        if (verbose) {
            ss << "\n[Hardware Execution Profile]\n";
            ss << "  Target SIMT Warp Width: 32 threads\n";
            ss << "  Thread Block Size:      256 threads\n";
            ss << "  Memory Coalescing:      128-byte aligned cache lines\n";
        }
    } catch (const std::exception& ex) {
        ss << "Exception during planning: " << ex.what() << "\n";
    }

    return ss.str();
}

} // namespace onyx::cli
