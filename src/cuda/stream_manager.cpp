#include "onyx/cuda/stream_manager.hpp"

namespace onyx::cuda {

CudaEvent::CudaEvent() {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    cudaEventCreate(&event_);
#else
    event_ = reinterpret_cast<cudaEvent_t>(1);
#endif
    cpu_timestamp_ = std::chrono::high_resolution_clock::now();
}

CudaEvent::~CudaEvent() {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    if (event_) {
        cudaEventDestroy(event_);
        event_ = nullptr;
    }
#endif
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept
    : event_(other.event_), cpu_timestamp_(other.cpu_timestamp_) {
    other.event_ = nullptr;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        if (event_) {
            cudaEventDestroy(event_);
        }
#endif
        event_ = other.event_;
        cpu_timestamp_ = other.cpu_timestamp_;
        other.event_ = nullptr;
    }
    return *this;
}

void CudaEvent::record([[maybe_unused]] cudaStream_t stream) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    cudaEventRecord(event_, stream);
#endif
    cpu_timestamp_ = std::chrono::high_resolution_clock::now();
}

void CudaEvent::synchronize() {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    cudaEventSynchronize(event_);
#endif
}

float CudaEvent::elapsed_millis(const CudaEvent& start, const CudaEvent& end) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start.event_, end.event_);
    return ms;
#else
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(
        end.cpu_timestamp_ - start.cpu_timestamp_).count();
    return static_cast<float>(diff) / 1000.0f;
#endif
}

StreamManager::StreamManager() {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    for (int i = 0; i < 4; ++i) {
        cudaStream_t s_high, s_norm, s_low;
        cudaStreamCreateWithPriority(&s_high, cudaStreamNonBlocking, -1);
        cudaStreamCreateWithPriority(&s_norm, cudaStreamNonBlocking, 0);
        cudaStreamCreateWithPriority(&s_low, cudaStreamNonBlocking, 1);
        high_streams_.push_back(s_high);
        normal_streams_.push_back(s_norm);
        low_streams_.push_back(s_low);
    }
#else
    normal_streams_.push_back(nullptr);
    high_streams_.push_back(nullptr);
    low_streams_.push_back(nullptr);
#endif
}

StreamManager::~StreamManager() {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    for (auto s : high_streams_) cudaStreamDestroy(s);
    for (auto s : normal_streams_) cudaStreamDestroy(s);
    for (auto s : low_streams_) cudaStreamDestroy(s);
#endif
}

cudaStream_t StreamManager::get_stream(StreamPriority priority) {
    const auto& list = (priority == StreamPriority::High) ? high_streams_
                     : (priority == StreamPriority::Low) ? low_streams_
                     : normal_streams_;
    if (list.empty()) return nullptr;
    size_t idx = (rr_counter_++) % list.size();
    return list[idx];
}

void StreamManager::synchronize_stream([[maybe_unused]] cudaStream_t stream) {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    if (stream) {
        cudaStreamSynchronize(stream);
    }
#endif
}

void StreamManager::synchronize_all() {
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    cudaDeviceSynchronize();
#endif
}

} // namespace onyx::cuda
