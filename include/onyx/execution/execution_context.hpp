#pragma once

#include "onyx/cuda/stream_manager.hpp"
#include "onyx/cuda/gpu_memory_pool.hpp"
#include <chrono>
#include <string>
#include <vector>

namespace onyx::execution {

struct OperatorMetrics {
    std::string operator_name;
    size_t input_rows{0};
    size_t output_rows{0};
    double kernel_time_us{0.0};
    size_t memory_bytes_used{0};

    [[nodiscard]] std::string to_string() const;
};

class ExecutionContext {
public:
    ExecutionContext(cudaStream_t stream = nullptr, cuda::GpuMemoryPool* pool = &cuda::GpuMemoryPool::global());
    ~ExecutionContext() = default;

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
    [[nodiscard]] cuda::GpuMemoryPool* memory_pool() noexcept { return pool_; }

    void record_metric(OperatorMetrics metric);
    [[nodiscard]] const std::vector<OperatorMetrics>& metrics() const noexcept { return metrics_; }
    [[nodiscard]] double total_kernel_time_us() const noexcept;

    void reset();

private:
    cudaStream_t stream_{nullptr};
    cuda::GpuMemoryPool* pool_{nullptr};
    std::vector<OperatorMetrics> metrics_;
};

} // namespace onyx::execution
