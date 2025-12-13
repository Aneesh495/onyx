#include "onyx/execution/execution_context.hpp"
#include <sstream>
#include <iomanip>

namespace onyx::execution {

std::string OperatorMetrics::to_string() const {
    std::ostringstream oss;
    oss << operator_name << " [in: " << input_rows << ", out: " << output_rows
        << ", time: " << std::fixed << std::setprecision(2) << kernel_time_us << " us"
        << ", mem: " << (memory_bytes_used / 1024) << " KB]";
    return oss.str();
}

ExecutionContext::ExecutionContext(cudaStream_t stream, cuda::GpuMemoryPool* pool)
    : stream_(stream), pool_(pool) {}

void ExecutionContext::record_metric(OperatorMetrics metric) {
    metrics_.push_back(std::move(metric));
}

double ExecutionContext::total_kernel_time_us() const noexcept {
    double total = 0.0;
    for (const auto& m : metrics_) {
        total += m.kernel_time_us;
    }
    return total;
}

void ExecutionContext::reset() {
    metrics_.clear();
}

} // namespace onyx::execution
