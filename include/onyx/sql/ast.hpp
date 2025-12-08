#pragma once

#include "onyx/common/types.hpp"
#include <string>
#include <vector>
#include <memory>

namespace onyx::sql {

enum class BinaryOp : uint8_t {
    Add = 0,
    Subtract,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    And,
    Or,
    Like
};

class Expr {
public:
    virtual ~Expr() = default;
    [[nodiscard]] virtual std::string to_string() const = 0;
};

class ColumnRefExpr : public Expr {
public:
    ColumnRefExpr(std::string column_name, std::string table_name = "")
        : column_name_(std::move(column_name)), table_name_(std::move(table_name)) {}

    [[nodiscard]] const std::string& column_name() const noexcept { return column_name_; }
    [[nodiscard]] const std::string& table_name() const noexcept { return table_name_; }

    [[nodiscard]] std::string to_string() const override;

private:
    std::string column_name_;
    std::string table_name_;
};

class LiteralExpr : public Expr {
public:
    explicit LiteralExpr(ScalarValue value) : value_(std::move(value)) {}

    [[nodiscard]] const ScalarValue& value() const noexcept { return value_; }

    [[nodiscard]] std::string to_string() const override;

private:
    ScalarValue value_;
};

class BinaryExpr : public Expr {
public:
    BinaryExpr(std::shared_ptr<Expr> left, BinaryOp op, std::shared_ptr<Expr> right)
        : left_(std::move(left)), op_(op), right_(std::move(right)) {}

    [[nodiscard]] std::shared_ptr<Expr> left() const noexcept { return left_; }
    [[nodiscard]] BinaryOp op() const noexcept { return op_; }
    [[nodiscard]] std::shared_ptr<Expr> right() const noexcept { return right_; }

    [[nodiscard]] std::string to_string() const override;

private:
    std::shared_ptr<Expr> left_;
    BinaryOp op_;
    std::shared_ptr<Expr> right_;
};

class FunctionCallExpr : public Expr {
public:
    FunctionCallExpr(std::string function_name, std::vector<std::shared_ptr<Expr>> arguments)
        : function_name_(std::move(function_name)), arguments_(std::move(arguments)) {}

    [[nodiscard]] const std::string& function_name() const noexcept { return function_name_; }
    [[nodiscard]] const std::vector<std::shared_ptr<Expr>>& arguments() const noexcept { return arguments_; }

    [[nodiscard]] std::string to_string() const override;

private:
    std::string function_name_;
    std::vector<std::shared_ptr<Expr>> arguments_;
};

class WindowExpr : public Expr {
public:
    WindowExpr(std::shared_ptr<FunctionCallExpr> function_call,
               std::vector<std::shared_ptr<Expr>> partition_by,
               std::vector<std::pair<std::shared_ptr<Expr>, bool>> order_by,
               uint32_t preceding_rows = 0)
        : function_call_(std::move(function_call)),
          partition_by_(std::move(partition_by)),
          order_by_(std::move(order_by)),
          preceding_rows_(preceding_rows) {}

    [[nodiscard]] std::shared_ptr<FunctionCallExpr> function_call() const noexcept { return function_call_; }
    [[nodiscard]] const std::vector<std::shared_ptr<Expr>>& partition_by() const noexcept { return partition_by_; }
    [[nodiscard]] const std::vector<std::pair<std::shared_ptr<Expr>, bool>>& order_by() const noexcept { return order_by_; }
    [[nodiscard]] uint32_t preceding_rows() const noexcept { return preceding_rows_; }

    [[nodiscard]] std::string to_string() const override;

private:
    std::shared_ptr<FunctionCallExpr> function_call_;
    std::vector<std::shared_ptr<Expr>> partition_by_;
    std::vector<std::pair<std::shared_ptr<Expr>, bool>> order_by_;
    uint32_t preceding_rows_{0};
};

struct SelectItem {
    std::shared_ptr<Expr> expr;
    std::string alias;
};

struct JoinClause {
    std::string table_name;
    std::string table_alias;
    std::shared_ptr<Expr> on_condition;
    bool is_left{false};
};

struct OrderByItem {
    std::shared_ptr<Expr> expr;
    bool ascending{true};
};

class SelectStatement {
public:
    SelectStatement() = default;

    std::vector<SelectItem> select_list;
    std::string from_table;
    std::string from_alias;
    std::vector<JoinClause> joins;
    std::shared_ptr<Expr> where_clause;
    std::vector<std::shared_ptr<Expr>> group_by;
    std::shared_ptr<Expr> having_clause;
    std::vector<OrderByItem> order_by;
    int64_t limit_count{-1};
    int64_t offset_count{0};

    [[nodiscard]] std::string to_string() const;
};

} // namespace onyx::sql
