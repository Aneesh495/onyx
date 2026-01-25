#include "onyx/execution/query_engine.hpp"
#include "onyx/sql/parser.hpp"
#include "onyx/planner/optimizer.hpp"
#include "onyx/execution/pipeline_executor.hpp"
#include "onyx/common/logger.hpp"
#include "onyx/common/error.hpp"
#include <sstream>
#include <chrono>

namespace onyx::execution {

std::string QueryResult::to_formatted_output(size_t max_rows) const {
    std::ostringstream oss;
    if (!status.is_ok()) {
        oss << "Query Error: " << status.message() << "\n";
        return oss.str();
    }

    if (table) {
        oss << table->to_formatted_string(max_rows);
    }

    oss << "\nExecution Summary:\n"
        << "  Total Latency: " << execution_time_ms << " ms\n"
        << "  Kernel Time:   " << kernel_time_us << " us\n";

    if (!metrics.empty()) {
        oss << "  Operator Breakdown:\n";
        for (const auto& m : metrics) {
            oss << "    * " << m.to_string() << "\n";
        }
    }
    return oss.str();
}

QueryEngine::QueryEngine(storage::Catalog& catalog) : catalog_(catalog) {}

QueryResult QueryEngine::execute_query(std::string_view sql) {
    auto wall_start = std::chrono::high_resolution_clock::now();
    QueryResult result;

    try {
        // Step 1: Parse SQL
        sql::Parser parser(sql);
        auto stmt = parser.parse_statement();
        if (!stmt) {
            result.status = Status::parser_error("Failed to parse statement");
            return result;
        }

        // Step 2: Logical Planning
        auto logical_plan = planner::LogicalPlanBuilder::build(*stmt, catalog_);
        result.logical_plan_str = logical_plan->to_string();

        // Step 3: Optimization & Physical Planning
        planner::Optimizer optimizer;
        auto physical_plan = optimizer.optimize(logical_plan);
        if (!physical_plan) {
            result.status = Status::planner_error("Failed to generate physical plan");
            return result;
        }
        result.physical_plan_str = physical_plan->to_string();

        // Step 4: Construct Physical Executor Tree
        auto executor = create_executor(physical_plan, catalog_);
        if (!executor) {
            result.status = Status::execution_error("Failed to create physical executor");
            return result;
        }

        // Step 5: Execute Pipeline
        cudaStream_t stream = stream_manager_.get_stream(cuda::StreamPriority::High);
        ExecutionContext ctx(stream);

        PipelineExecutor pipeline(std::move(executor));
        result.table = pipeline.execute(ctx);
        result.metrics = ctx.metrics();
        result.kernel_time_us = ctx.total_kernel_time_us();
        result.status = Status::ok();

    } catch (const OnyxException& ex) {
        result.status = ex.status();
    } catch (const std::exception& ex) {
        result.status = Status::execution_error(ex.what());
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    return result;
}

std::string QueryEngine::explain_query(std::string_view sql) {
    std::ostringstream oss;
    try {
        sql::Parser parser(sql);
        auto stmt = parser.parse_statement();
        auto logical_plan = planner::LogicalPlanBuilder::build(*stmt, catalog_);
        planner::Optimizer optimizer;
        auto physical_plan = optimizer.optimize(logical_plan);

        oss << "=== Logical Query Plan ===\n"
            << logical_plan->to_string()
            << "\n=== Optimized GPU Physical Plan ===\n"
            << physical_plan->to_string();
    } catch (const std::exception& ex) {
        oss << "Explain Error: " << ex.what() << "\n";
    }
    return oss.str();
}

} // namespace onyx::execution
