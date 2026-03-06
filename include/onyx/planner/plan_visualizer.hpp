#pragma once

#include "onyx/planner/logical_plan.hpp"
#include "onyx/planner/physical_plan.hpp"
#include <string>
#include <memory>

namespace onyx::planner {

class PlanVisualizer {
public:
    // Format a logical plan tree into an ASCII representation
    static std::string format_logical_plan(const LogicalOperator& plan, int indent_level = 0);

    // Format a physical execution plan into an ASCII execution DAG
    static std::string format_physical_plan(const PhysicalOperator& plan, int indent_level = 0);
};

} // namespace onyx::planner
