#include "onyx/planner/plan_visualizer.hpp"
#include <sstream>

namespace onyx::planner {

static std::string make_indent(int level) {
    std::string s;
    for (int i = 0; i < level; ++i) {
        s += (i == level - 1) ? "|-- " : "    ";
    }
    return s;
}

std::string PlanVisualizer::format_logical_plan(const LogicalOperator& plan, int indent_level) {
    std::ostringstream ss;
    ss << make_indent(indent_level) << plan.to_string(0) << "\n";

    for (const auto& child : plan.children()) {
        if (child) {
            ss << format_logical_plan(*child, indent_level + 1);
        }
    }
    return ss.str();
}

std::string PlanVisualizer::format_physical_plan(const PhysicalOperator& plan, int indent_level) {
    std::ostringstream ss;
    ss << make_indent(indent_level) << plan.to_string(0) << "\n";

    for (const auto& child : plan.children()) {
        if (child) {
            ss << format_physical_plan(*child, indent_level + 1);
        }
    }
    return ss.str();
}

} // namespace onyx::planner
