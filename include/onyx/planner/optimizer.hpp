#pragma once

#include "onyx/planner/logical_plan.hpp"
#include "onyx/planner/physical_plan.hpp"
#include <memory>

namespace onyx::planner {

class Optimizer {
public:
    Optimizer() = default;

    // Optimize logical plan into physical plan
    [[nodiscard]] std::shared_ptr<PhysicalOperator> optimize(std::shared_ptr<LogicalOperator> logical_plan);

private:
    std::shared_ptr<LogicalOperator> push_down_predicates(std::shared_ptr<LogicalOperator> op);
    std::shared_ptr<LogicalOperator> prune_columns(std::shared_ptr<LogicalOperator> op);
    std::shared_ptr<PhysicalOperator> translate_to_physical(std::shared_ptr<LogicalOperator> op);
    std::shared_ptr<PhysicalOperator> try_fuse_pipeline(std::shared_ptr<LogicalOperator> op);
};

} // namespace onyx::planner
