#pragma once

#include "onyx/planner/logical_plan.hpp"
#include "onyx/kernels/filter.cuh"
#include <string>
#include <vector>
#include <memory>

namespace onyx::planner {

enum class PhysicalOpType : uint8_t {
    Scan = 0,
    Filter,
    Project,
    HashJoin,
    Aggregate,
    Sort,
    Window,
    Limit,
    FusedPipeline
};

class PhysicalOperator {
public:
    virtual ~PhysicalOperator() = default;
    [[nodiscard]] virtual PhysicalOpType op_type() const noexcept = 0;
    [[nodiscard]] virtual std::shared_ptr<storage::Schema> schema() const noexcept = 0;
    [[nodiscard]] virtual std::string to_string(int indent = 0) const = 0;

    void add_child(std::shared_ptr<PhysicalOperator> child) { children_.push_back(std::move(child)); }
    [[nodiscard]] const std::vector<std::shared_ptr<PhysicalOperator>>& children() const noexcept { return children_; }

protected:
    std::vector<std::shared_ptr<PhysicalOperator>> children_;
};

class PhysicalScan : public PhysicalOperator {
public:
    PhysicalScan(std::string table_name, std::shared_ptr<storage::Schema> schema, std::vector<std::string> projected_cols)
        : table_name_(std::move(table_name)), schema_(std::move(schema)), projected_cols_(std::move(projected_cols)) {}

    [[nodiscard]] PhysicalOpType op_type() const noexcept override { return PhysicalOpType::Scan; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override { return schema_; }
    [[nodiscard]] const std::string& table_name() const noexcept { return table_name_; }
    [[nodiscard]] const std::vector<std::string>& projected_cols() const noexcept { return projected_cols_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    std::string table_name_;
    std::shared_ptr<storage::Schema> schema_;
    std::vector<std::string> projected_cols_;
};

class PhysicalFilter : public PhysicalOperator {
public:
    PhysicalFilter(std::shared_ptr<PhysicalOperator> input, std::shared_ptr<sql::Expr> predicate)
        : predicate_(std::move(predicate)) {
        add_child(std::move(input));
    }

    [[nodiscard]] PhysicalOpType op_type() const noexcept override { return PhysicalOpType::Filter; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override { return children_[0]->schema(); }
    [[nodiscard]] std::shared_ptr<sql::Expr> predicate() const noexcept { return predicate_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    std::shared_ptr<sql::Expr> predicate_;
};

class PhysicalProject : public PhysicalOperator {
public:
    PhysicalProject(std::shared_ptr<PhysicalOperator> input, std::vector<sql::SelectItem> items, std::shared_ptr<storage::Schema> schema)
        : items_(std::move(items)), schema_(std::move(schema)) {
        add_child(std::move(input));
    }

    [[nodiscard]] PhysicalOpType op_type() const noexcept override { return PhysicalOpType::Project; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override { return schema_; }
    [[nodiscard]] const std::vector<sql::SelectItem>& items() const noexcept { return items_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    std::vector<sql::SelectItem> items_;
    std::shared_ptr<storage::Schema> schema_;
};

class PhysicalHashJoin : public PhysicalOperator {
public:
    PhysicalHashJoin(std::shared_ptr<PhysicalOperator> build,
                     std::shared_ptr<PhysicalOperator> probe,
                     std::shared_ptr<sql::Expr> on_condition,
                     bool is_left_join)
        : on_condition_(std::move(on_condition)), is_left_join_(is_left_join) {
        add_child(std::move(build));
        add_child(std::move(probe));
    }

    [[nodiscard]] PhysicalOpType op_type() const noexcept override { return PhysicalOpType::HashJoin; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override;
    [[nodiscard]] std::shared_ptr<sql::Expr> on_condition() const noexcept { return on_condition_; }
    [[nodiscard]] bool is_left_join() const noexcept { return is_left_join_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    std::shared_ptr<sql::Expr> on_condition_;
    bool is_left_join_{false};
};

class PhysicalAggregate : public PhysicalOperator {
public:
    PhysicalAggregate(std::shared_ptr<PhysicalOperator> input,
                      std::vector<std::shared_ptr<sql::Expr>> group_exprs,
                      std::vector<std::shared_ptr<sql::FunctionCallExpr>> agg_exprs,
                      std::shared_ptr<storage::Schema> schema)
        : group_exprs_(std::move(group_exprs)), agg_exprs_(std::move(agg_exprs)), schema_(std::move(schema)) {
        add_child(std::move(input));
    }

    [[nodiscard]] PhysicalOpType op_type() const noexcept override { return PhysicalOpType::Aggregate; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override { return schema_; }
    [[nodiscard]] const std::vector<std::shared_ptr<sql::Expr>>& group_exprs() const noexcept { return group_exprs_; }
    [[nodiscard]] const std::vector<std::shared_ptr<sql::FunctionCallExpr>>& agg_exprs() const noexcept { return agg_exprs_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    std::vector<std::shared_ptr<sql::Expr>> group_exprs_;
    std::vector<std::shared_ptr<sql::FunctionCallExpr>> agg_exprs_;
    std::shared_ptr<storage::Schema> schema_;
};

class PhysicalSort : public PhysicalOperator {
public:
    PhysicalSort(std::shared_ptr<PhysicalOperator> input, std::vector<sql::OrderByItem> order_by)
        : order_by_(std::move(order_by)) {
        add_child(std::move(input));
    }

    [[nodiscard]] PhysicalOpType op_type() const noexcept override { return PhysicalOpType::Sort; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override { return children_[0]->schema(); }
    [[nodiscard]] const std::vector<sql::OrderByItem>& order_by() const noexcept { return order_by_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    std::vector<sql::OrderByItem> order_by_;
};

class PhysicalLimit : public PhysicalOperator {
public:
    PhysicalLimit(std::shared_ptr<PhysicalOperator> input, int64_t limit, int64_t offset = 0)
        : limit_(limit), offset_(offset) {
        add_child(std::move(input));
    }

    [[nodiscard]] PhysicalOpType op_type() const noexcept override { return PhysicalOpType::Limit; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override { return children_[0]->schema(); }
    [[nodiscard]] int64_t limit() const noexcept { return limit_; }
    [[nodiscard]] int64_t offset() const noexcept { return offset_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    int64_t limit_;
    int64_t offset_;
};

class PhysicalFusedPipeline : public PhysicalOperator {
public:
    PhysicalFusedPipeline(std::string table_name,
                          std::string filter_col,
                          ScalarValue filter_literal,
                          kernels::CompareOp filter_op,
                          std::string col_a,
                          std::string col_b,
                          std::shared_ptr<storage::Schema> schema)
        : table_name_(std::move(table_name)),
          filter_col_(std::move(filter_col)),
          filter_literal_(std::move(filter_literal)),
          filter_op_(filter_op),
          col_a_(std::move(col_a)),
          col_b_(std::move(col_b)),
          schema_(std::move(schema)) {}

    [[nodiscard]] PhysicalOpType op_type() const noexcept override { return PhysicalOpType::FusedPipeline; }
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const noexcept override { return schema_; }
    [[nodiscard]] const std::string& table_name() const noexcept { return table_name_; }
    [[nodiscard]] const std::string& filter_col() const noexcept { return filter_col_; }
    [[nodiscard]] const ScalarValue& filter_literal() const noexcept { return filter_literal_; }
    [[nodiscard]] kernels::CompareOp filter_op() const noexcept { return filter_op_; }
    [[nodiscard]] const std::string& col_a() const noexcept { return col_a_; }
    [[nodiscard]] const std::string& col_b() const noexcept { return col_b_; }

    [[nodiscard]] std::string to_string(int indent = 0) const override;

private:
    std::string table_name_;
    std::string filter_col_;
    ScalarValue filter_literal_;
    kernels::CompareOp filter_op_;
    std::string col_a_;
    std::string col_b_;
    std::shared_ptr<storage::Schema> schema_;
};

} // namespace onyx::planner
