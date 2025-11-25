#include "onyx/cuda/cuda_graph.hpp"
#include "onyx/common/logger.hpp"

namespace onyx::cuda {

CudaGraphExecutor::CudaGraphExecutor() = default;

CudaGraphExecutor::~CudaGraphExecutor() {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    if (instance_) {
        cudaGraphExecDestroy(instance_);
        instance_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
#endif
}

CudaGraphExecutor::CudaGraphExecutor(CudaGraphExecutor&& other) noexcept
    : graph_(other.graph_), instance_(other.instance_),
      is_capturing_(other.is_capturing_), is_instantiated_(other.is_instantiated_) {
    other.graph_ = nullptr;
    other.instance_ = nullptr;
    other.is_capturing_ = false;
    other.is_instantiated_ = false;
}

CudaGraphExecutor& CudaGraphExecutor::operator=(CudaGraphExecutor&& other) noexcept {
    if (this != &other) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        if (instance_) cudaGraphExecDestroy(instance_);
        if (graph_) cudaGraphDestroy(graph_);
#endif
        graph_ = other.graph_;
        instance_ = other.instance_;
        is_capturing_ = other.is_capturing_;
        is_instantiated_ = other.is_instantiated_;
        other.graph_ = nullptr;
        other.instance_ = nullptr;
        other.is_capturing_ = false;
        other.is_instantiated_ = false;
    }
    return *this;
}

void CudaGraphExecutor::begin_capture([[maybe_unused]] cudaStream_t stream) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    if (stream) {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        is_capturing_ = true;
    }
#else
    is_capturing_ = true;
#endif
}

void CudaGraphExecutor::end_capture([[maybe_unused]] cudaStream_t stream) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    if (stream && is_capturing_) {
        cudaStreamEndCapture(stream, &graph_);
        cudaGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
        is_capturing_ = false;
        is_instantiated_ = true;
    }
#else
    is_capturing_ = false;
    is_instantiated_ = true;
#endif
}

void CudaGraphExecutor::launch([[maybe_unused]] cudaStream_t stream) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    if (instance_) {
        cudaGraphLaunch(instance_, stream);
    }
#else
    // In emulation mode, kernels are executed synchronously
#endif
}

} // namespace onyx::cuda
