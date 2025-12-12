#include "onyx/planner/cost_model.hpp"
#include <sstream>
#include <iomanip>

namespace onyx::planner {

std::string PlanCost::to_string() const {
    std::ostringstream oss;
    oss << "PlanCost { latency: " << std::fixed << std::setprecision(2) << estimated_latency_us << " us"
        << ", memory: " << (estimated_gpu_memory_bytes / 1024) << " KB"
        << ", pcie: " << (estimated_pcie_transfer_bytes / 1024) << " KB"
        << ", intensity: " << arithmetic_intensity << " FLOP/B }";
    return oss.str();
}

PlanCost CostModel::estimate_cost(const LogicalOperator& op, size_t input_rows) {
    PlanCost cost;
    switch (op.op_type()) {
        case LogicalOpType::Scan: {
            size_t bytes = input_rows * 8; // 8 bytes per numeric column estimate
            cost.estimated_pcie_transfer_bytes = bytes;
            cost.estimated_gpu_memory_bytes = bytes;
            cost.estimated_latency_us = static_cast<double>(bytes) / (25.0 * 1024 * 1024); // 25 GB/s PCIe
            cost.arithmetic_intensity = 0.1;
            break;
        }
        case LogicalOpType::Filter: {
            cost.estimated_gpu_memory_bytes = input_rows * 4;
            cost.estimated_latency_us = static_cast<double>(input_rows) / 100000.0; // 100M rows/s filter
            cost.arithmetic_intensity = 1.0;
            break;
        }
        case LogicalOpType::Join: {
            size_t table_bytes = input_rows * 16;
            cost.estimated_gpu_memory_bytes = table_bytes * 2;
            cost.estimated_latency_us = static_cast<double>(input_rows) / 50000.0; // 50M rows/s hash join
            cost.arithmetic_intensity = 4.0;
            break;
        }
        case LogicalOpType::Aggregate: {
            cost.estimated_gpu_memory_bytes = 1024 * 64;
            cost.estimated_latency_us = static_cast<double>(input_rows) / 200000.0;
            cost.arithmetic_intensity = 2.0;
            break;
        }
        case LogicalOpType::Sort: {
            cost.estimated_gpu_memory_bytes = input_rows * 16;
            cost.estimated_latency_us = static_cast<double>(input_rows) / 30000.0; // Radix sort
            cost.arithmetic_intensity = 8.0;
            break;
        }
        default:
            cost.estimated_latency_us = 10.0;
            break;
    }
    return cost;
}

} // namespace onyx::planner
