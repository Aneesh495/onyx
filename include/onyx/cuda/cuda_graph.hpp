#pragma once

#include "onyx/cuda/cuda_runtime_compat.hpp"
#include <functional>
#include <string>

namespace onyx::cuda {

class CudaGraphExecutor {
public:
    CudaGraphExecutor();
    ~CudaGraphExecutor();

    CudaGraphExecutor(const CudaGraphExecutor&) = delete;
    CudaGraphExecutor& operator=(const CudaGraphExecutor&) = delete;
    CudaGraphExecutor(CudaGraphExecutor&& other) noexcept;
    CudaGraphExecutor& operator=(CudaGraphExecutor&& other) noexcept;

    // Capture a sequence of stream operations into a CUDA Graph
    void begin_capture(cudaStream_t stream);
    void end_capture(cudaStream_t stream);

    // Launch the instantiated graph
    void launch(cudaStream_t stream = nullptr);

    [[nodiscard]] bool is_instantiated() const noexcept { return is_instantiated_; }

private:
    cudaGraph_t graph_{nullptr};
    cudaGraphExec_t instance_{nullptr};
    bool is_capturing_{false};
    bool is_instantiated_{false};
};

} // namespace onyx::cuda
