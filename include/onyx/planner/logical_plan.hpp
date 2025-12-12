#pragma once

#include "onyx/sql/ast.hpp"
#include "onyx/storage/catalog.hpp"
#include <string>
#include <vector>
#include <memory>

namespace onyx::planner {

enum class LogicalOpType : uint8_t {
    Scan = 0,
    Filter,
    Project,
    Join,
    Aggregate,
    Sort,
    Window,
    Limit
};

class LogicalOperator {
public:
    virtual ~LogicalOperator() = default;
    [[nodiscard]] virtual LogicalOpType op_type() const noexcept = 0;
    [[nodiscard]] virtual std::shared_ptr<storage::Schema> schema() const noexcept = 0;
    [[nodiscard]] virtual std::string to_string(int indent = 0) const = 0;

    void add_child(std::shared_ptr<LogicalOperator> child) { children_.push_back(std::move(child)); }
    [[nodiscard]] const std::vector<std::shared_ptr<LogicalOperator>>& children() const noexcept { return children_; }

protected:
    std::vector<std::shared_ptr<LogicalOperator>> children_;
};

class LogicalScan : public LogicalOperator {
public:
    LogicalScan(std::string table_name, std::shared_ptr<storage::Schema> schema, std::vector<std::string> projected_columns = {})
        : table_name_(std::move(table_name)), schema_(std::move(schema)), projected_columns_(std::move(projected_columns)) {}

    [[nodiscard]] LogicalOpType op_type() const noexcept override { return LogicalOpType::Scan; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override { return schema_; }
    [[nodiscard]] const std::string& table_name() const noexcept { return table_name_; }
    [[nodiscard]] const std::vector<std::string>& projected_columns() const noexcept { return projected_columns_; }

    void set_projected_columns(std::vector<std::string> cols) { projected_columns_ = std::move(cols); }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    std::string table_name_;
    std::shared_ptr<storage::Schema> schema_;
    std::vector<std::string> projected_columns_;
};

class LogicalFilter : public LogicalOperator {
public:
    LogicalFilter(std::shared_ptr<LogicalOperator> input, std::shared_ptr<sql::Expr> predicate)
        : predicate_(std::move(predicate)) {
        add_child(std::move(input));
    }

    [[nodiscard]] LogicalOpType op_type() const noexcept override { return LogicalOpType::Filter; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override { return children_[0]->schema(); }
    [[nodiscard]] std::shared_ptr<sql::Expr> predicate() const noexcept { return predicate_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    std::shared_ptr<sql::Expr> predicate_;
};

class LogicalProject : public LogicalOperator {
public:
    LogicalProject(std::shared_ptr<LogicalOperator> input, std::vector<sql::SelectItem> items, std::shared_ptr<storage::Schema> schema)
        : items_(std::move(items)), schema_(std::move(schema)) {
        add_child(std::move(input));
    }

    [[nodiscard]] LogicalOpType op_type() const noexcept override { return LogicalOpType::Project; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override { return schema_; }
    [[nodiscard]] const std::vector<sql::SelectItem>& items() const noexcept { return items_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    std::vector<sql::SelectItem> items_;
    std::shared_ptr<storage::Schema> schema_;
};

class LogicalJoin : public LogicalOperator {
public:
    LogicalJoin(std::shared_ptr<LogicalOperator> left,
                std::shared_ptr<LogicalOperator> right,
                std::shared_ptr<sql::Expr> on_condition,
                bool is_left_join)
        : on_condition_(std::move(on_condition)), is_left_join_(is_left_join) {
        add_child(std::move(left));
        add_child(std::move(right));
    }

    [[nodiscard]] LogicalOpType op_type() const noexcept override { return LogicalOpType::Join; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override;
    [[nodiscard]] std::shared_ptr<sql::Expr> on_condition() const noexcept { return on_condition_; }
    [[nodiscard]] bool is_left_join() const noexcept { return is_left_join_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    std::shared_ptr<sql::Expr> on_condition_;
    bool is_left_join_{false};
};

class LogicalAggregate : public LogicalOperator {
public:
    LogicalAggregate(std::shared_ptr<LogicalOperator> input,
                     std::vector<std::shared_ptr<sql::Expr>> group_exprs,
                     std::vector<std::shared_ptr<sql::FunctionCallExpr>> agg_exprs,
                     std::shared_ptr<storage::Schema> schema)
        : group_exprs_(std::move(group_exprs)), agg_exprs_(std::move(agg_exprs)), schema_(std::move(schema)) {
        add_child(std::move(input));
    }

    [[nodiscard]] LogicalOpType op_type() const noexcept override { return LogicalOpType::Aggregate; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override { return schema_; }
    [[nodiscard]] const std::vector<std::shared_ptr<sql::Expr>>& group_exprs() const noexcept { return group_exprs_; }
    [[nodiscard]] const std::vector<std::shared_ptr<sql::FunctionCallExpr>>& agg_exprs() const noexcept { return agg_exprs_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    std::vector<std::shared_ptr<sql::Expr>> group_exprs_;
    std::vector<std::shared_ptr<sql::FunctionCallExpr>> agg_exprs_;
    std::shared_ptr<storage::Schema> schema_;
};

class LogicalSort : public LogicalOperator {
public:
    LogicalSort(std::shared_ptr<LogicalOperator> input, std::vector<sql::OrderByItem> order_by)
        : order_by_(std::move(order_by)) {
        add_child(std::move(input));
    }

    [[nodiscard]] LogicalOpType op_type() const noexcept override { return LogicalOpType::Sort; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override { return children_[0]->schema(); }
    [[nodiscard]] const std::vector<sql::OrderByItem>& order_by() const noexcept { return order_by_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    std::vector<sql::OrderByItem> order_by_;
};

class LogicalLimit : public LogicalOperator {
public:
    LogicalLimit(std::shared_ptr<LogicalOperator> input, int64_t limit, int64_t offset = 0)
        : limit_(limit), offset_(offset) {
        add_child(std::move(input));
    }

    [[nodiscard]] LogicalOpType op_type() const noexcept override { return LogicalOpType::Limit; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override { return children_[0]->schema(); }
    [[nodiscard]] int64_t limit() const noexcept { return limit_; }
    [[nodiscard]] int64_t offset() const noexcept { return offset_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    int64_t limit_;
    int64_t offset_;
};

// Builds a logical plan from AST
class LogicalPlanBuilder {
public:
    static std::shared_ptr<LogicalOperator> build(
        const sql::SelectStatement& stmt,
        const storage::Catalog& catalog = storage::Catalog::instance());
};

} // namespace onyx::planner
