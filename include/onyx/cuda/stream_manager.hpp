#pragma once

#include "onyx/cuda/cuda_runtime_compat.hpp"
#include <vector>
#include <memory>
#include <chrono>

namespace onyx::cuda {

enum class StreamPriority : int {
    High = -1,
    Normal = 0,
    Low = 1
};

class CudaEvent {
public:
    CudaEvent();
    ~CudaEvent();

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& other) noexcept;
    CudaEvent& operator=(CudaEvent&& other) noexcept;

    void record(cudaStream_t stream = nullptr);
    void synchronize();

    // Returns elapsed time between start and end event in milliseconds
    static float elapsed_millis(const CudaEvent& start, const CudaEvent& end);

    [[nodiscard]] cudaEvent_t handle() const noexcept { return event_; }

private:
    cudaEvent_t event_{nullptr};
    std::chrono::high_resolution_clock::time_point cpu_timestamp_;
};

class StreamManager {
public:
    static StreamManager& instance() {
        static StreamManager mgr;
        return mgr;
    }

    StreamManager();
    ~StreamManager();

    [[nodiscard]] cudaStream_t get_stream(StreamPriority priority = StreamPriority::Normal);
    void synchronize_stream(cudaStream_t stream);
    void synchronize_all();

private:
    std::vector<cudaStream_t> high_streams_;
    std::vector<cudaStream_t> normal_streams_;
    std::vector<cudaStream_t> low_streams_;
    size_t rr_counter_{0};
};

} // namespace onyx::cuda
