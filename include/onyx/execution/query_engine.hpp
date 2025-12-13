#pragma once

#include "onyx/storage/catalog.hpp"
#include "onyx/execution/execution_context.hpp"
#include <string>
#include <memory>
#include <vector>

namespace onyx::execution {

struct QueryResult {
    Status status;
    std::shared_ptr<storage::Table> table;
    double execution_time_ms{0.0};
    double kernel_time_us{0.0};
    std::vector<OperatorMetrics> metrics;
    std::string logical_plan_str;
    std::string physical_plan_str;

    [[nodiscard]] std::string to_formatted_output(size_t max_rows = 50) const;
};

class QueryEngine {
public:
    static QueryEngine& instance() {
        static QueryEngine engine;
        return engine;
    }

    explicit QueryEngine(storage::Catalog& catalog = storage::Catalog::instance());
    ~QueryEngine() = default;

    // Execute SQL query
    [[nodiscard]] QueryResult execute_query(std::string_view sql);

    // Explain query without executing
    [[nodiscard]] std::string explain_query(std::string_view sql);

    [[nodiscard]] storage::Catalog& catalog() noexcept { return catalog_; }

private:
    storage::Catalog& catalog_;
    cuda::StreamManager stream_manager_;
};

} // namespace onyx::execution
