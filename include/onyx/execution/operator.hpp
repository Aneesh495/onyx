#pragma once

#include "onyx/planner/physical_plan.hpp"
#include "onyx/storage/column_batch.hpp"
#include "onyx/storage/catalog.hpp"
#include "onyx/execution/execution_context.hpp"
#include <memory>
#include <vector>

namespace onyx::execution {

class PhysicalExecutor {
public:
    virtual ~PhysicalExecutor() = default;
    virtual void open(ExecutionContext& ctx) = 0;
    [[nodiscard]] virtual std::shared_ptr<storage::ColumnBatch> next(ExecutionContext& ctx) = 0;
    virtual void close(ExecutionContext& ctx) = 0;
    [[nodiscard]] virtual std::shared_ptr<storage::Schema> schema() const = 0;
};

class ScanExecutor : public PhysicalExecutor {
public:
    ScanExecutor(std::shared_ptr<planner::PhysicalScan> plan, storage::Catalog& catalog = storage::Catalog::instance());

    void open(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::ColumnBatch> next(ExecutionContext& ctx) override;
    void close(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const override { return plan_->schema(); }

private:
    std::shared_ptr<planner::PhysicalScan> plan_;
    storage::Catalog& catalog_;
    std::shared_ptr<storage::Table> table_;
    size_t batch_cursor_{0};
};

class FilterExecutor : public PhysicalExecutor {
public:
    FilterExecutor(std::shared_ptr<planner::PhysicalFilter> plan, std::unique_ptr<PhysicalExecutor> child);

    void open(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::ColumnBatch> next(ExecutionContext& ctx) override;
    void close(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const override { return plan_->schema(); }

private:
    std::shared_ptr<planner::PhysicalFilter> plan_;
    std::unique_ptr<PhysicalExecutor> child_;
};

class ProjectExecutor : public PhysicalExecutor {
public:
    ProjectExecutor(std::shared_ptr<planner::PhysicalProject> plan, std::unique_ptr<PhysicalExecutor> child);

    void open(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::ColumnBatch> next(ExecutionContext& ctx) override;
    void close(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const override { return plan_->schema(); }

private:
    std::shared_ptr<planner::PhysicalProject> plan_;
    std::unique_ptr<PhysicalExecutor> child_;
};

class HashJoinExecutor : public PhysicalExecutor {
public:
    HashJoinExecutor(std::shared_ptr<planner::PhysicalHashJoin> plan,
                     std::unique_ptr<PhysicalExecutor> build_child,
                     std::unique_ptr<PhysicalExecutor> probe_child);

    void open(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::ColumnBatch> next(ExecutionContext& ctx) override;
    void close(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const override { return plan_->schema(); }

private:
    std::shared_ptr<planner::PhysicalHashJoin> plan_;
    std::unique_ptr<PhysicalExecutor> build_child_;
    std::unique_ptr<PhysicalExecutor> probe_child_;
    std::shared_ptr<storage::ColumnBatch> build_table_data_;
    bool build_completed_{false};
};

class AggregateExecutor : public PhysicalExecutor {
public:
    AggregateExecutor(std::shared_ptr<planner::PhysicalAggregate> plan, std::unique_ptr<PhysicalExecutor> child);

    void open(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::ColumnBatch> next(ExecutionContext& ctx) override;
    void close(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const override { return plan_->schema(); }

private:
    std::shared_ptr<planner::PhysicalAggregate> plan_;
    std::unique_ptr<PhysicalExecutor> child_;
    bool emitted_{false};
};

class SortExecutor : public PhysicalExecutor {
public:
    SortExecutor(std::shared_ptr<planner::PhysicalSort> plan, std::unique_ptr<PhysicalExecutor> child);

    void open(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::ColumnBatch> next(ExecutionContext& ctx) override;
    void close(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const override { return plan_->schema(); }

private:
    std::shared_ptr<planner::PhysicalSort> plan_;
    std::unique_ptr<PhysicalExecutor> child_;
    std::shared_ptr<storage::ColumnBatch> sorted_batch_;
    bool emitted_{false};
};

class LimitExecutor : public PhysicalExecutor {
public:
    LimitExecutor(std::shared_ptr<planner::PhysicalLimit> plan, std::unique_ptr<PhysicalExecutor> child);

    void open(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::ColumnBatch> next(ExecutionContext& ctx) override;
    void close(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const override { return plan_->schema(); }

private:
    std::shared_ptr<planner::PhysicalLimit> plan_;
    std::unique_ptr<PhysicalExecutor> child_;
    size_t rows_emitted_{0};
};

class FusedPipelineExecutor : public PhysicalExecutor {
public:
    FusedPipelineExecutor(std::shared_ptr<planner::PhysicalFusedPipeline> plan, storage::Catalog& catalog = storage::Catalog::instance());

    void open(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::ColumnBatch> next(ExecutionContext& ctx) override;
    void close(ExecutionContext& ctx) override;
    [[nodiscard]] std::shared_ptr<storage::Schema> schema() const override { return plan_->schema(); }

private:
    std::shared_ptr<planner::PhysicalFusedPipeline> plan_;
    storage::Catalog& catalog_;
    std::shared_ptr<storage::Table> table_;
    bool executed_{false};
};

// Factory method to construct an executor tree from physical plan
std::unique_ptr<PhysicalExecutor> create_executor(
    const std::shared_ptr<planner::PhysicalOperator>& plan,
    storage::Catalog& catalog = storage::Catalog::instance());

} // namespace onyx::execution
