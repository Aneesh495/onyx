#include "onyx/planner/logical_plan.hpp"
#include "onyx/common/error.hpp"
#include <sstream>

namespace onyx::planner {

static std::string make_indent(int indent) {
    return std::string(indent * 2, ' ');
}

std::string LogicalScan::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "LogicalScan: table=" << table_name_ << " [";
    for (size_t i = 0; i < projected_columns_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << projected_columns_[i];
    }
    oss << "]\n";
    return oss.str();
}

std::string LogicalFilter::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "LogicalFilter: " << predicate_->to_string() << "\n";
    if (!children_.empty()) {
        oss << children_[0]->to_string(indent + 1);
    }
    return oss.str();
}

std::string LogicalProject::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "LogicalProject: [";
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

std::shared_ptr<storage::Schema> LogicalJoin::schema() const noexcept {
    std::vector<storage::Field> fields;
    if (!children_.empty() && children_[0]->schema()) {
        for (const auto& f : children_[0]->schema()->fields()) fields.push_back(f);
    }
    if (children_.size() > 1 && children_[1]->schema()) {
        for (const auto& f : children_[1]->schema()->fields()) fields.push_back(f);
    }
    return std::make_shared<storage::Schema>(std::move(fields));
}

std::string LogicalJoin::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "LogicalJoin: (" << (is_left_join_ ? "LEFT" : "INNER") << ") ON "
        << (on_condition_ ? on_condition_->to_string() : "true") << "\n";
    for (const auto& child : children_) {
        oss << child->to_string(indent + 1);
    }
    return oss.str();
}

std::string LogicalAggregate::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "LogicalAggregate: groups=[";
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

std::string LogicalSort::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "LogicalSort: [";
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

std::string LogicalLimit::to_string(int indent) const {
    std::ostringstream oss;
    oss << make_indent(indent) << "LogicalLimit: limit=" << limit_ << " offset=" << offset_ << "\n";
    if (!children_.empty()) {
        oss << children_[0]->to_string(indent + 1);
    }
    return oss.str();
}

std::shared_ptr<LogicalOperator> LogicalPlanBuilder::build(
    const sql::SelectStatement& stmt,
    const storage::Catalog& catalog)
{
    if (stmt.from_table.empty()) {
        throw PlannerException("SELECT without FROM is not supported");
    }

    auto table = catalog.get_table(stmt.from_table);
    if (!table) {
        throw PlannerException("Table not found in catalog: " + stmt.from_table);
    }

    std::vector<std::string> all_cols;
    for (const auto& field : table->schema()->fields()) {
        all_cols.push_back(field.name);
    }

    std::shared_ptr<LogicalOperator> plan = std::make_shared<LogicalScan>(
        stmt.from_table, table->schema(), all_cols);

    // JOIN
    for (const auto& j : stmt.joins) {
        auto right_table = catalog.get_table(j.table_name);
        if (!right_table) {
            throw PlannerException("Joined table not found: " + j.table_name);
        }
        std::vector<std::string> right_cols;
        for (const auto& f : right_table->schema()->fields()) right_cols.push_back(f.name);
        auto right_scan = std::make_shared<LogicalScan>(j.table_name, right_table->schema(), right_cols);
        plan = std::make_shared<LogicalJoin>(plan, right_scan, j.on_condition, j.is_left);
    }

    // WHERE
    if (stmt.where_clause) {
        plan = std::make_shared<LogicalFilter>(plan, stmt.where_clause);
    }

    // AGGREGATION / GROUP BY
    bool has_aggregates = !stmt.group_by.empty();
    std::vector<std::shared_ptr<sql::FunctionCallExpr>> aggs;
    for (const auto& item : stmt.select_list) {
        if (auto func = std::dynamic_pointer_cast<sql::FunctionCallExpr>(item.expr)) {
            has_aggregates = true;
            aggs.push_back(func);
        }
    }

    if (has_aggregates) {
        std::vector<storage::Field> agg_fields;
        for (size_t i = 0; i < stmt.group_by.size(); ++i) {
            agg_fields.push_back(storage::Field{"group_" + std::to_string(i), DataType(TypeId::INT64)});
        }
        for (size_t i = 0; i < aggs.size(); ++i) {
            agg_fields.push_back(storage::Field{aggs[i]->function_name(), DataType(TypeId::FLOAT64)});
        }
        plan = std::make_shared<LogicalAggregate>(
            plan, stmt.group_by, aggs, std::make_shared<storage::Schema>(std::move(agg_fields)));
    } else {
        // PROJECT
        std::vector<storage::Field> proj_fields;
        for (const auto& item : stmt.select_list) {
            std::string col_name = item.alias.empty() ? item.expr->to_string() : item.alias;
            proj_fields.push_back(storage::Field{col_name, DataType(TypeId::FLOAT64)});
        }
        plan = std::make_shared<LogicalProject>(
            plan, stmt.select_list, std::make_shared<storage::Schema>(std::move(proj_fields)));
    }

    // ORDER BY
    if (!stmt.order_by.empty()) {
        plan = std::make_shared<LogicalSort>(plan, stmt.order_by);
    }

    // LIMIT
    if (stmt.limit_count >= 0) {
        plan = std::make_shared<LogicalLimit>(plan, stmt.limit_count, stmt.offset_count);
    }

    return plan;
}

} // namespace onyx::planner
