#include "onyx/planner/optimizer.hpp"
#include "onyx/common/logger.hpp"

namespace onyx::planner {

std::shared_ptr<PhysicalOperator> Optimizer::optimize(std::shared_ptr<LogicalOperator> logical_plan) {
    if (!logical_plan) return nullptr;

    // Try operator fusion first
    auto fused = try_fuse_pipeline(logical_plan);
    if (fused) {
        ONYX_LOG_INFO("Optimizer fused query into PhysicalFusedPipeline");
        return fused;
    }

    // Pass 1: Push down predicates
    auto pushed = push_down_predicates(logical_plan);

    // Pass 2: Prune unused columns
    auto pruned = prune_columns(pushed);

    // Pass 3: Translate to physical operators
    return translate_to_physical(pruned);
}

std::shared_ptr<PhysicalOperator> Optimizer::try_fuse_pipeline(std::shared_ptr<LogicalOperator> op) {
    // Check if pattern is Aggregate(Filter(Scan))
    if (op->op_type() != LogicalOpType::Aggregate) return nullptr;
    auto agg = std::dynamic_pointer_cast<LogicalAggregate>(op);
    if (!agg || agg->children().empty() || !agg->group_exprs().empty()) return nullptr;
    if (agg->agg_exprs().size() != 1 || agg->agg_exprs()[0]->function_name() != "sum") return nullptr;

    auto child = agg->children()[0];
    if (child->op_type() != LogicalOpType::Filter) return nullptr;
    auto filter = std::dynamic_pointer_cast<LogicalFilter>(child);
    if (!filter || filter->children().empty()) return nullptr;

    auto scan_child = filter->children()[0];
    if (scan_child->op_type() != LogicalOpType::Scan) return nullptr;
    auto scan = std::dynamic_pointer_cast<LogicalScan>(scan_child);
    if (!scan) return nullptr;

    // Check filter predicate
    auto bin_expr = std::dynamic_pointer_cast<sql::BinaryExpr>(filter->predicate());
    if (!bin_expr) return nullptr;

    auto col_ref = std::dynamic_pointer_cast<sql::ColumnRefExpr>(bin_expr->left());
    auto lit = std::dynamic_pointer_cast<sql::LiteralExpr>(bin_expr->right());
    if (!col_ref || !lit) return nullptr;

    kernels::CompareOp cmp_op = kernels::CompareOp::Equal;
    switch (bin_expr->op()) {
        case sql::BinaryOp::Equal: cmp_op = kernels::CompareOp::Equal; break;
        case sql::BinaryOp::NotEqual: cmp_op = kernels::CompareOp::NotEqual; break;
        case sql::BinaryOp::LessThan: cmp_op = kernels::CompareOp::LessThan; break;
        case sql::BinaryOp::LessThanOrEqual: cmp_op = kernels::CompareOp::LessThanOrEqual; break;
        case sql::BinaryOp::GreaterThan: cmp_op = kernels::CompareOp::GreaterThan; break;
        case sql::BinaryOp::GreaterThanOrEqual: cmp_op = kernels::CompareOp::GreaterThanOrEqual; break;
        default: return nullptr;
    }

    // Inspect aggregate argument: sum(a * b) or sum(a)
    if (agg->agg_exprs()[0]->arguments().empty()) return nullptr;
    auto agg_arg = agg->agg_exprs()[0]->arguments()[0];

    std::string col_a, col_b;
    if (auto mul_expr = std::dynamic_pointer_cast<sql::BinaryExpr>(agg_arg)) {
        if (mul_expr->op() == sql::BinaryOp::Multiply) {
            auto a_ref = std::dynamic_pointer_cast<sql::ColumnRefExpr>(mul_expr->left());
            auto b_ref = std::dynamic_pointer_cast<sql::ColumnRefExpr>(mul_expr->right());
            if (a_ref && b_ref) {
                col_a = a_ref->column_name();
                col_b = b_ref->column_name();
            }
        }
    } else if (auto single_ref = std::dynamic_pointer_cast<sql::ColumnRefExpr>(agg_arg)) {
        col_a = single_ref->column_name();
    }

    if (col_a.empty()) return nullptr;

    return std::make_shared<PhysicalFusedPipeline>(
        scan->table_name(),
        col_ref->column_name(),
        lit->value(),
        cmp_op,
        col_a,
        col_b,
        agg->schema());
}

std::shared_ptr<LogicalOperator> Optimizer::push_down_predicates(std::shared_ptr<LogicalOperator> op) {
    if (!op) return nullptr;
    return op;
}

std::shared_ptr<LogicalOperator> Optimizer::prune_columns(std::shared_ptr<LogicalOperator> op) {
    if (!op) return nullptr;
    return op;
}

std::shared_ptr<PhysicalOperator> Optimizer::translate_to_physical(std::shared_ptr<LogicalOperator> op) {
    if (!op) return nullptr;

    std::vector<std::shared_ptr<PhysicalOperator>> phys_children;
    for (const auto& child : op->children()) {
        phys_children.push_back(translate_to_physical(child));
    }

    switch (op->op_type()) {
        case LogicalOpType::Scan: {
            auto scan = std::dynamic_pointer_cast<LogicalScan>(op);
            return std::make_shared<PhysicalScan>(scan->table_name(), scan->schema(), scan->projected_columns());
        }
        case LogicalOpType::Filter: {
            auto filter = std::dynamic_pointer_cast<LogicalFilter>(op);
            return std::make_shared<PhysicalFilter>(phys_children[0], filter->predicate());
        }
        case LogicalOpType::Project: {
            auto proj = std::dynamic_pointer_cast<LogicalProject>(op);
            return std::make_shared<PhysicalProject>(phys_children[0], proj->items(), proj->schema());
        }
        case LogicalOpType::Join: {
            auto join = std::dynamic_pointer_cast<LogicalJoin>(op);
            return std::make_shared<PhysicalHashJoin>(phys_children[0], phys_children[1], join->on_condition(), join->is_left_join());
        }
        case LogicalOpType::Aggregate: {
            auto agg = std::dynamic_pointer_cast<LogicalAggregate>(op);
            return std::make_shared<PhysicalAggregate>(phys_children[0], agg->group_exprs(), agg->agg_exprs(), agg->schema());
        }
        case LogicalOpType::Sort: {
            auto sort = std::dynamic_pointer_cast<LogicalSort>(op);
            return std::make_shared<PhysicalSort>(phys_children[0], sort->order_by());
        }
        case LogicalOpType::Limit: {
            auto limit = std::dynamic_pointer_cast<LogicalLimit>(op);
            return std::make_shared<PhysicalLimit>(phys_children[0], limit->limit(), limit->offset());
        }
        default:
            return nullptr;
    }
}

} // namespace onyx::planner
