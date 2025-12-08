#include "onyx/sql/ast.hpp"
#include <sstream>

namespace onyx::sql {

std::string ColumnRefExpr::to_string() const {
    if (!table_name_.empty()) {
        return table_name_ + "." + column_name_;
    }
    return column_name_;
}

std::string LiteralExpr::to_string() const {
    return value_.to_string();
}

std::string BinaryExpr::to_string() const {
    std::string op_str;
    switch (op_) {
        case BinaryOp::Add: op_str = "+"; break;
        case BinaryOp::Subtract: op_str = "-"; break;
        case BinaryOp::Multiply: op_str = "*"; break;
        case BinaryOp::Divide: op_str = "/"; break;
        case BinaryOp::Equal: op_str = "="; break;
        case BinaryOp::NotEqual: op_str = "!="; break;
        case BinaryOp::LessThan: op_str = "<"; break;
        case BinaryOp::LessThanOrEqual: op_str = "<="; break;
        case BinaryOp::GreaterThan: op_str = ">"; break;
        case BinaryOp::GreaterThanOrEqual: op_str = ">="; break;
        case BinaryOp::And: op_str = "AND"; break;
        case BinaryOp::Or: op_str = "OR"; break;
        case BinaryOp::Like: op_str = "LIKE"; break;
    }
    return "(" + left_->to_string() + " " + op_str + " " + right_->to_string() + ")";
}

std::string FunctionCallExpr::to_string() const {
    std::ostringstream oss;
    oss << function_name_ << "(";
    for (size_t i = 0; i < arguments_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << arguments_[i]->to_string();
    }
    oss << ")";
    return oss.str();
}

std::string WindowExpr::to_string() const {
    std::ostringstream oss;
    oss << function_call_->to_string() << " OVER (";
    if (!partition_by_.empty()) {
        oss << "PARTITION BY ";
        for (size_t i = 0; i < partition_by_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << partition_by_[i]->to_string();
        }
        oss << " ";
    }
    if (!order_by_.empty()) {
        oss << "ORDER BY ";
        for (size_t i = 0; i < order_by_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << order_by_[i].first->to_string() << (order_by_[i].second ? " ASC" : " DESC");
        }
    }
    if (preceding_rows_ > 0) {
        oss << " ROWS BETWEEN " << preceding_rows_ << " PRECEDING AND CURRENT ROW";
    }
    oss << ")";
    return oss.str();
}

std::string SelectStatement::to_string() const {
    std::ostringstream oss;
    oss << "SELECT ";
    for (size_t i = 0; i < select_list.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << select_list[i].expr->to_string();
        if (!select_list[i].alias.empty()) {
            oss << " AS " << select_list[i].alias;
        }
    }
    if (!from_table.empty()) {
        oss << " FROM " << from_table;
        if (!from_alias.empty()) oss << " " << from_alias;
    }
    for (const auto& j : joins) {
        oss << (j.is_left ? " LEFT JOIN " : " INNER JOIN ") << j.table_name;
        if (!j.table_alias.empty()) oss << " " << j.table_alias;
        if (j.on_condition) oss << " ON " << j.on_condition->to_string();
    }
    if (where_clause) {
        oss << " WHERE " << where_clause->to_string();
    }
    if (!group_by.empty()) {
        oss << " GROUP BY ";
        for (size_t i = 0; i < group_by.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << group_by[i]->to_string();
        }
    }
    if (having_clause) {
        oss << " HAVING " << having_clause->to_string();
    }
    if (!order_by.empty()) {
        oss << " ORDER BY ";
        for (size_t i = 0; i < order_by.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << order_by[i].expr->to_string() << (order_by[i].ascending ? " ASC" : " DESC");
        }
    }
    if (limit_count >= 0) {
        oss << " LIMIT " << limit_count;
    }
    if (offset_count > 0) {
        oss << " OFFSET " << offset_count;
    }
    return oss.str();
}

} // namespace onyx::sql
