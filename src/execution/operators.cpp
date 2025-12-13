#include "onyx/execution/operator.hpp"
#include "onyx/kernels/filter.cuh"
#include "onyx/kernels/radix_sort.cuh"
#include "onyx/kernels/hash_join.cuh"
#include "onyx/kernels/aggregate.cuh"
#include "onyx/kernels/fused_pipeline.cuh"
#include "onyx/common/error.hpp"
#include <chrono>

namespace onyx::execution {

// ScanExecutor
ScanExecutor::ScanExecutor(std::shared_ptr<planner::PhysicalScan> plan, storage::Catalog& catalog)
    : plan_(std::move(plan)), catalog_(catalog) {}

void ScanExecutor::open([[maybe_unused]] ExecutionContext& ctx) {
    table_ = catalog_.get_table(plan_->table_name());
    batch_cursor_ = 0;
}

std::shared_ptr<storage::ColumnBatch> ScanExecutor::next([[maybe_unused]] ExecutionContext& ctx) {
    if (!table_ || batch_cursor_ >= table_->num_batches()) {
        return nullptr;
    }
    auto batch = table_->batch(batch_cursor_++);
    return batch;
}

void ScanExecutor::close([[maybe_unused]] ExecutionContext& ctx) {
    batch_cursor_ = 0;
}

// FilterExecutor
FilterExecutor::FilterExecutor(std::shared_ptr<planner::PhysicalFilter> plan, std::unique_ptr<PhysicalExecutor> child)
    : plan_(std::move(plan)), child_(std::move(child)) {}

void FilterExecutor::open(ExecutionContext& ctx) {
    if (child_) child_->open(ctx);
}

std::shared_ptr<storage::ColumnBatch> FilterExecutor::next(ExecutionContext& ctx) {
    if (!child_) return nullptr;
    auto input_batch = child_->next(ctx);
    if (!input_batch || input_batch->num_rows() == 0) return nullptr;

    auto bin_expr = std::dynamic_pointer_cast<sql::BinaryExpr>(plan_->predicate());
    if (!bin_expr) return input_batch;

    auto col_ref = std::dynamic_pointer_cast<sql::ColumnRefExpr>(bin_expr->left());
    auto lit = std::dynamic_pointer_cast<sql::LiteralExpr>(bin_expr->right());
    if (!col_ref || !lit) return input_batch;

    auto col = input_batch->column(col_ref->column_name());
    if (!col) return input_batch;

    kernels::CompareOp cmp_op = kernels::CompareOp::Equal;
    switch (bin_expr->op()) {
        case sql::BinaryOp::Equal:              cmp_op = kernels::CompareOp::Equal; break;
        case sql::BinaryOp::NotEqual:           cmp_op = kernels::CompareOp::NotEqual; break;
        case sql::BinaryOp::LessThan:           cmp_op = kernels::CompareOp::LessThan; break;
        case sql::BinaryOp::LessThanOrEqual:    cmp_op = kernels::CompareOp::LessThanOrEqual; break;
        case sql::BinaryOp::GreaterThan:        cmp_op = kernels::CompareOp::GreaterThan; break;
        case sql::BinaryOp::GreaterThanOrEqual: cmp_op = kernels::CompareOp::GreaterThanOrEqual; break;
        default: break;
    }

    size_t n = input_batch->num_rows();
    std::vector<uint32_t> matched_ids(n);
    uint32_t match_count = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    if (col->type().id == TypeId::FLOAT64) {
        kernels::launch_filter_column<double>(
            col->data<double>(), col->null_bitmap(), n, lit->value().as_double(), cmp_op,
            matched_ids.data(), &match_count, ctx.stream());
    } else if (col->type().id == TypeId::INT64) {
        kernels::launch_filter_column<int64_t>(
            col->data<int64_t>(), col->null_bitmap(), n, lit->value().as_int64(), cmp_op,
            matched_ids.data(), &match_count, ctx.stream());
    } else if (col->type().id == TypeId::INT32) {
        kernels::launch_filter_column<int32_t>(
            col->data<int32_t>(), col->null_bitmap(), n, static_cast<int32_t>(lit->value().as_int64()), cmp_op,
            matched_ids.data(), &match_count, ctx.stream());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    ctx.record_metric(OperatorMetrics{
        .operator_name = "PhysicalFilter",
        .input_rows = n,
        .output_rows = match_count,
        .kernel_time_us = us,
        .memory_bytes_used = match_count * sizeof(uint32_t)
    });

    if (match_count == 0) return nullptr;
    return input_batch->filter(matched_ids.data(), match_count, ctx.stream());
}

void FilterExecutor::close(ExecutionContext& ctx) {
    if (child_) child_->close(ctx);
}

// ProjectExecutor
ProjectExecutor::ProjectExecutor(std::shared_ptr<planner::PhysicalProject> plan, std::unique_ptr<PhysicalExecutor> child)
    : plan_(std::move(plan)), child_(std::move(child)) {}

void ProjectExecutor::open(ExecutionContext& ctx) {
    if (child_) child_->open(ctx);
}

std::shared_ptr<storage::ColumnBatch> ProjectExecutor::next(ExecutionContext& ctx) {
    if (!child_) return nullptr;
    auto input_batch = child_->next(ctx);
    if (!input_batch || input_batch->num_rows() == 0) return nullptr;

    auto result_batch = std::make_shared<storage::ColumnBatch>(plan_->schema(), input_batch->num_rows());

    for (size_t i = 0; i < plan_->items().size(); ++i) {
        const auto& item = plan_->items()[i];
        if (auto col_ref = std::dynamic_pointer_cast<sql::ColumnRefExpr>(item.expr)) {
            if (col_ref->column_name() == "*") {
                return input_batch;
            }
            auto in_col = input_batch->column(col_ref->column_name());
            if (in_col) {
                result_batch->set_column(i, in_col);
            }
        }
    }
    return result_batch;
}

void ProjectExecutor::close(ExecutionContext& ctx) {
    if (child_) child_->close(ctx);
}

// HashJoinExecutor
HashJoinExecutor::HashJoinExecutor(std::shared_ptr<planner::PhysicalHashJoin> plan,
                                   std::unique_ptr<PhysicalExecutor> build_child,
                                   std::unique_ptr<PhysicalExecutor> probe_child)
    : plan_(std::move(plan)), build_child_(std::move(build_child)), probe_child_(std::move(probe_child)) {}

void HashJoinExecutor::open(ExecutionContext& ctx) {
    if (build_child_) build_child_->open(ctx);
    if (probe_child_) probe_child_->open(ctx);
    build_completed_ = false;
}

std::shared_ptr<storage::ColumnBatch> HashJoinExecutor::next(ExecutionContext& ctx) {
    if (!build_completed_) {
        // Ingest build table
        while (auto batch = build_child_->next(ctx)) {
            if (!build_table_data_) {
                build_table_data_ = batch;
            }
        }
        build_completed_ = true;
    }

    if (!build_table_data_ || build_table_data_->num_rows() == 0) return nullptr;

    auto probe_batch = probe_child_->next(ctx);
    if (!probe_batch || probe_batch->num_rows() == 0) return nullptr;

    // Join keys
    size_t num_build = build_table_data_->num_rows();
    size_t num_probe = probe_batch->num_rows();

    std::vector<uint32_t> out_build_ids(num_probe);
    std::vector<uint32_t> out_probe_ids(num_probe);
    uint32_t match_count = 0;

    auto build_col = build_table_data_->column(0);
    auto probe_col = probe_batch->column(0);

    if (build_col && probe_col && build_col->type().id == TypeId::INT32) {
        kernels::launch_hash_join<int32_t>(
            build_col->data<int32_t>(), num_build,
            probe_col->data<int32_t>(), num_probe,
            out_build_ids.data(), out_probe_ids.data(), &match_count, ctx.stream());
    }

    if (match_count == 0) return nullptr;

    // Construct joined batch
    auto res = std::make_shared<storage::ColumnBatch>(plan_->schema(), match_count);
    return res;
}

void HashJoinExecutor::close(ExecutionContext& ctx) {
    if (build_child_) build_child_->close(ctx);
    if (probe_child_) probe_child_->close(ctx);
    build_completed_ = false;
}

// AggregateExecutor
AggregateExecutor::AggregateExecutor(std::shared_ptr<planner::PhysicalAggregate> plan, std::unique_ptr<PhysicalExecutor> child)
    : plan_(std::move(plan)), child_(std::move(child)) {}

void AggregateExecutor::open(ExecutionContext& ctx) {
    if (child_) child_->open(ctx);
    emitted_ = false;
}

std::shared_ptr<storage::ColumnBatch> AggregateExecutor::next(ExecutionContext& ctx) {
    if (emitted_ || !child_) return nullptr;

    double accumulated_sum = 0.0;
    uint64_t accumulated_count = 0;
    size_t total_input_rows = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (auto batch = child_->next(ctx)) {
        if (batch->num_rows() == 0) continue;
        total_input_rows += batch->num_rows();

        if (plan_->agg_exprs().empty()) continue;
        auto agg_func = plan_->agg_exprs()[0];

        // Evaluate aggregate argument column
        std::shared_ptr<storage::Column> col = nullptr;
        if (!agg_func->arguments().empty()) {
            if (auto col_ref = std::dynamic_pointer_cast<sql::ColumnRefExpr>(agg_func->arguments()[0])) {
                col = batch->column(col_ref->column_name());
            }
        }
        if (!col && batch->num_columns() > 0) {
            col = batch->column(0);
        }

        if (col) {
            double batch_sum = 0.0;
            uint64_t batch_count = 0;

            if (col->type().id == TypeId::FLOAT64) {
                kernels::launch_scalar_aggregate<double>(
                    col->data<double>(), col->null_bitmap(), batch->num_rows(),
                    &batch_sum, &batch_count, ctx.stream());
            } else if (col->type().id == TypeId::INT64) {
                kernels::launch_scalar_aggregate<int64_t>(
                    col->data<int64_t>(), col->null_bitmap(), batch->num_rows(),
                    &batch_sum, &batch_count, ctx.stream());
            } else if (col->type().id == TypeId::INT32) {
                kernels::launch_scalar_aggregate<int32_t>(
                    col->data<int32_t>(), col->null_bitmap(), batch->num_rows(),
                    &batch_sum, &batch_count, ctx.stream());
            }
            accumulated_sum += batch_sum;
            accumulated_count += batch_count;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    ctx.record_metric(OperatorMetrics{
        .operator_name = "PhysicalAggregate",
        .input_rows = total_input_rows,
        .output_rows = 1,
        .kernel_time_us = us,
        .memory_bytes_used = 128
    });

    auto result_batch = std::make_shared<storage::ColumnBatch>(plan_->schema(), 1);
    if (!plan_->agg_exprs().empty() && result_batch->column(0)) {
        std::string fname = plan_->agg_exprs()[0]->function_name();
        for (char& c : fname) c = std::tolower(c);
        if (fname == "count") {
            result_batch->column(0)->append_scalar(ScalarValue::make_float64(static_cast<double>(accumulated_count)));
        } else if (fname == "avg") {
            double avg = accumulated_count > 0 ? (accumulated_sum / static_cast<double>(accumulated_count)) : 0.0;
            result_batch->column(0)->append_scalar(ScalarValue::make_float64(avg));
        } else {
            result_batch->column(0)->append_scalar(ScalarValue::make_float64(accumulated_sum));
        }
    }

    emitted_ = true;
    return result_batch;
}

void AggregateExecutor::close(ExecutionContext& ctx) {
    if (child_) child_->close(ctx);
    emitted_ = false;
}

// SortExecutor
SortExecutor::SortExecutor(std::shared_ptr<planner::PhysicalSort> plan, std::unique_ptr<PhysicalExecutor> child)
    : plan_(std::move(plan)), child_(std::move(child)) {}

void SortExecutor::open(ExecutionContext& ctx) {
    if (child_) child_->open(ctx);
    emitted_ = false;
}

std::shared_ptr<storage::ColumnBatch> SortExecutor::next(ExecutionContext& ctx) {
    if (emitted_ || !child_) return nullptr;

    // Collect all batches
    std::shared_ptr<storage::ColumnBatch> full_batch = nullptr;
    while (auto b = child_->next(ctx)) {
        if (!full_batch) full_batch = b;
    }

    if (!full_batch || full_batch->num_rows() == 0) return nullptr;

    if (!plan_->order_by().empty()) {
        auto item = plan_->order_by()[0];
        if (auto col_ref = std::dynamic_pointer_cast<sql::ColumnRefExpr>(item.expr)) {
            auto sort_col = full_batch->column(col_ref->column_name());
            if (sort_col && sort_col->type().id == TypeId::INT32) {
                std::vector<uint32_t> row_indices(full_batch->num_rows());
                for (size_t i = 0; i < row_indices.size(); ++i) row_indices[i] = static_cast<uint32_t>(i);

                kernels::launch_radix_sort<int32_t, uint32_t>(
                    sort_col->data<int32_t>(), row_indices.data(), full_batch->num_rows(), item.ascending, ctx.stream());

                full_batch = full_batch->filter(row_indices.data(), row_indices.size(), ctx.stream());
            }
        }
    }

    emitted_ = true;
    return full_batch;
}

void SortExecutor::close(ExecutionContext& ctx) {
    if (child_) child_->close(ctx);
    emitted_ = false;
}

// LimitExecutor
LimitExecutor::LimitExecutor(std::shared_ptr<planner::PhysicalLimit> plan, std::unique_ptr<PhysicalExecutor> child)
    : plan_(std::move(plan)), child_(std::move(child)) {}

void LimitExecutor::open(ExecutionContext& ctx) {
    if (child_) child_->open(ctx);
    rows_emitted_ = 0;
}

std::shared_ptr<storage::ColumnBatch> LimitExecutor::next(ExecutionContext& ctx) {
    if (!child_ || (plan_->limit() >= 0 && rows_emitted_ >= static_cast<size_t>(plan_->limit()))) {
        return nullptr;
    }
    auto batch = child_->next(ctx);
    if (!batch) return nullptr;

    size_t allowed = static_cast<size_t>(plan_->limit()) - rows_emitted_;
    if (batch->num_rows() > allowed) {
        std::vector<uint32_t> ids(allowed);
        for (size_t i = 0; i < allowed; ++i) ids[i] = static_cast<uint32_t>(i);
        batch = batch->filter(ids.data(), allowed, ctx.stream());
    }

    rows_emitted_ += batch->num_rows();
    return batch;
}

void LimitExecutor::close(ExecutionContext& ctx) {
    if (child_) child_->close(ctx);
    rows_emitted_ = 0;
}

// FusedPipelineExecutor
FusedPipelineExecutor::FusedPipelineExecutor(std::shared_ptr<planner::PhysicalFusedPipeline> plan, storage::Catalog& catalog)
    : plan_(std::move(plan)), catalog_(catalog) {}

void FusedPipelineExecutor::open([[maybe_unused]] ExecutionContext& ctx) {
    table_ = catalog_.get_table(plan_->table_name());
    executed_ = false;
}

std::shared_ptr<storage::ColumnBatch> FusedPipelineExecutor::next(ExecutionContext& ctx) {
    if (executed_ || !table_) return nullptr;

    double total_sum = 0.0;
    uint64_t total_count = 0;
    size_t total_rows = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto& batch : table_->batches()) {
        total_rows += batch->num_rows();
        auto filter_col = batch->column(plan_->filter_col());
        auto col_a = batch->column(plan_->col_a());
        auto col_b = plan_->col_b().empty() ? nullptr : batch->column(plan_->col_b());

        if (filter_col && col_a && filter_col->type().id == TypeId::INT32) {
            double batch_sum = 0.0;
            uint64_t batch_count = 0;
            const double* b_ptr = col_b ? col_b->data<double>() : nullptr;

            kernels::launch_fused_filter_project_sum<int32_t, double>(
                filter_col->data<int32_t>(),
                static_cast<int32_t>(plan_->filter_literal().as_int64()),
                plan_->filter_op(),
                col_a->data<double>(),
                b_ptr,
                batch->num_rows(),
                &batch_sum,
                &batch_count,
                ctx.stream());

            total_sum += batch_sum;
            total_count += batch_count;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    ctx.record_metric(OperatorMetrics{
        .operator_name = "PhysicalFusedPipeline",
        .input_rows = total_rows,
        .output_rows = 1,
        .kernel_time_us = us,
        .memory_bytes_used = 128
    });

    auto res = std::make_shared<storage::ColumnBatch>(plan_->schema(), 1);
    if (res->column(0)) {
        res->column(0)->append_scalar(ScalarValue::make_float64(total_sum));
    }

    executed_ = true;
    return res;
}

void FusedPipelineExecutor::close([[maybe_unused]] ExecutionContext& ctx) {
    executed_ = false;
}

std::unique_ptr<PhysicalExecutor> create_executor(
    const std::shared_ptr<planner::PhysicalOperator>& plan,
    storage::Catalog& catalog)
{
    if (!plan) return nullptr;

    switch (plan->op_type()) {
        case planner::PhysicalOpType::Scan: {
            auto scan = std::dynamic_pointer_cast<planner::PhysicalScan>(plan);
            return std::make_unique<ScanExecutor>(scan, catalog);
        }
        case planner::PhysicalOpType::Filter: {
            auto filter = std::dynamic_pointer_cast<planner::PhysicalFilter>(plan);
            auto child = create_executor(filter->children()[0], catalog);
            return std::make_unique<FilterExecutor>(filter, std::move(child));
        }
        case planner::PhysicalOpType::Project: {
            auto proj = std::dynamic_pointer_cast<planner::PhysicalProject>(plan);
            auto child = create_executor(proj->children()[0], catalog);
            return std::make_unique<ProjectExecutor>(proj, std::move(child));
        }
        case planner::PhysicalOpType::HashJoin: {
            auto join = std::dynamic_pointer_cast<planner::PhysicalHashJoin>(plan);
            auto left = create_executor(join->children()[0], catalog);
            auto right = create_executor(join->children()[1], catalog);
            return std::make_unique<HashJoinExecutor>(join, std::move(left), std::move(right));
        }
        case planner::PhysicalOpType::Aggregate: {
            auto agg = std::dynamic_pointer_cast<planner::PhysicalAggregate>(plan);
            auto child = create_executor(agg->children()[0], catalog);
            return std::make_unique<AggregateExecutor>(agg, std::move(child));
        }
        case planner::PhysicalOpType::Sort: {
            auto sort = std::dynamic_pointer_cast<planner::PhysicalSort>(plan);
            auto child = create_executor(sort->children()[0], catalog);
            return std::make_unique<SortExecutor>(sort, std::move(child));
        }
        case planner::PhysicalOpType::Limit: {
            auto limit = std::dynamic_pointer_cast<planner::PhysicalLimit>(plan);
            auto child = create_executor(limit->children()[0], catalog);
            return std::make_unique<LimitExecutor>(limit, std::move(child));
        }
        case planner::PhysicalOpType::FusedPipeline: {
            auto fused = std::dynamic_pointer_cast<planner::PhysicalFusedPipeline>(plan);
            return std::make_unique<FusedPipelineExecutor>(fused, catalog);
        }
        default:
            return nullptr;
    }
}

} // namespace onyx::execution
