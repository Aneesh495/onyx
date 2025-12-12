#include "onyx/planner/physical_plan.hpp"
#include <sstream>

namespace onyx::planner {

static std::string make_indent(int indent) {
    return std::string(indent * 2, ' ');
}

std::string PhysicalScan::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "PhysicalScan [GPU Memory Coalesced]: table=" << table_name_ << " cols=[";
    for (size_t i = 0; i < projected_cols_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << projected_cols_[i];
    }
    oss << "]\n";
    return oss.str();
}

std::string PhysicalFilter::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "PhysicalFilter [GPU Warp Voting & Compaction]: " << predicate_->to_string() << "\n";
    if (!children_.empty()) {
        oss << children_[0]->to_string(indent + 1);
    }
    return oss.str();
}

std::string PhysicalProject::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "PhysicalProject [GPU Vectorized SIMT]: [";
    for (size_t i = 0; i < items_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << items_[i].expr->to_string();
        if (!items_[i].alias.empty()) oss << " AS " << items_[i].alias;
    }
    oss << "]\n";
    if (!children_.empty()) {
        oss << children_[0]->to_string(indent + 1);
    }
    return oss.str();
}

std::shared_ptr<storage::Schema> PhysicalHashJoin::schema() const noexcept {
    std::vector<storage::Field> fields;
    if (!children_.empty() && children_[0]->schema()) {
        for (const auto& f : children_[0]->schema()->fields()) fields.push_back(f);
    }
    if (children_.size() > 1 && children_[1]->schema()) {
        for (const auto& f : children_[1]->schema()->fields()) fields.push_back(f);
    }
    return std::make_shared<storage::Schema>(std::move(fields));
}

std::string PhysicalHashJoin::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "PhysicalHashJoin [Lock-Free atomicCAS & Warp-Cooperative Probe]: "
        << (is_left_join_ ? "LEFT" : "INNER") << " ON " << (on_condition_ ? on_condition_->to_string() : "true") << "\n";
    for (const auto& c : children_) {
        oss << c->to_string(indent + 1);
    }
    return oss.str();
}

std::string PhysicalAggregate::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "PhysicalAggregate [Hierarchical Warp & Block Shared Memory Reductions]: groups=[";
    for (size_t i = 0; i < group_exprs_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << group_exprs_[i]->to_string();
    }
    oss << "] aggs=[";
    for (size_t i = 0; i < agg_exprs_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << agg_exprs_[i]->to_string();
    }
    oss << "]\n";
    if (!children_.empty()) {
        oss << children_[0]->to_string(indent + 1);
    }
    return oss.str();
}

std::string PhysicalSort::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "PhysicalSort [GPU Multi-Pass LSD Radix Sort]: [";
    for (size_t i = 0; i < order_by_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << order_by_[i].expr->to_string() << (order_by_[i].ascending ? " ASC" : " DESC");
    }
    oss << "]\n";
    if (!children_.empty()) {
        oss << children_[0]->to_string(indent + 1);
    }
    return oss.str();
}

std::string PhysicalLimit::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "PhysicalLimit: limit=" << limit_ << " offset=" << offset_ << "\n";
    if (!children_.empty()) {
        oss << children_[0]->to_string(indent + 1);
    }
    return oss.str();
}

std::string PhysicalFusedPipeline::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "PhysicalFusedPipeline [Fused Scan-Filter-Project-Sum, Zero HBM Round-Trips]: "
        << "table=" << table_name_ << " filter=(" << filter_col_ << ") col_a=" << col_a_ << " col_b=" << col_b_ << "\n";
    return oss.str();
}

} // namespace onyx::planner
