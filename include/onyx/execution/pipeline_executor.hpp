#pragma once

#include "onyx/execution/operator.hpp"
#include <memory>

namespace onyx::execution {

class PipelineExecutor {
public:
    explicit PipelineExecutor(std::unique_ptr<PhysicalExecutor> root);
    ~PipelineExecutor() = default;

    // Execute the complete pipeline and collect all result batches into a single Table
    [[nodiscard]] std::shared_ptr<storage::Table> execute(ExecutionContext& ctx, std::string result_table_name = "query_result");

private:
    std::unique_ptr<PhysicalExecutor> root_;
};

} // namespace onyx::execution
