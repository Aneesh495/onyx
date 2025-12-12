#pragma once

#include "onyx/planner/logical_plan.hpp"
#include <cstdint>
#include <string>

namespace onyx::planner {

struct PlanCost {
    double estimated_latency_us{0.0};
    size_t estimated_gpu_memory_bytes{0};
    size_t estimated_pcie_transfer_bytes{0};
    double arithmetic_intensity{0.0}; // FLOPs per byte

    [[nodiscard]] std::string to_string() const;
};

class CostModel {
public:
    static PlanCost estimate_cost(const LogicalOperator& op, size_t input_rows = 1000000);
};

} // namespace onyx::planner
