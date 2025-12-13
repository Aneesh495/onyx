#include "onyx/execution/pipeline_executor.hpp"

namespace onyx::execution {

PipelineExecutor::PipelineExecutor(std::unique_ptr<PhysicalExecutor> root)
    : root_(std::move(root)) {}

std::shared_ptr<storage::Table> PipelineExecutor::execute(ExecutionContext& ctx, std::string result_table_name) {
    if (!root_) return nullptr;

    root_->open(ctx);
    auto schema = root_->schema();
    auto result_table = std::make_shared<storage::Table>(std::move(result_table_name), schema);

    while (auto batch = root_->next(ctx)) {
        if (batch->num_rows() > 0) {
            result_table->append_batch(batch);
        }
    }

    root_->close(ctx);
    return result_table;
}

} // namespace onyx::execution
