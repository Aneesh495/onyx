#pragma once

#include "onyx/cuda/cuda_runtime_compat.hpp"
#include <cstdint>
#include <functional>
#include <vector>
#include <memory>

namespace onyx::cuda {

#if !defined(ONYX_HAS_CUDA) || ONYX_HAS_CUDA == 0

struct Dim3 {
    uint32_t x{1};
    uint32_t y{1};
    uint32_t z{1};

    constexpr Dim3() = default;
    constexpr Dim3(uint32_t x_) : x(x_), y(1), z(1) {}
    constexpr Dim3(uint32_t x_, uint32_t y_) : x(x_), y(y_), z(1) {}
    constexpr Dim3(uint32_t x_, uint32_t y_, uint32_t z_) : x(x_), y(y_), z(z_) {}

    [[nodiscard]] constexpr size_t total_threads() const noexcept {
        return static_cast<size_t>(x) * y * z;
    }
};

using dim3 = Dim3;

struct SimtThreadContext {
    Dim3 thread_idx{0, 0, 0};
    Dim3 block_idx{0, 0, 0};
    Dim3 block_dim{1, 1, 1};
    Dim3 grid_dim{1, 1, 1};
};

extern thread_local SimtThreadContext g_simt_thread_ctx;

class SimtEngine {
public:
    static SimtEngine& instance() {
        static SimtEngine engine;
        return engine;
    }

    void set_num_workers(size_t workers) noexcept {
        num_workers_ = workers;
    }

    [[nodiscard]] size_t get_num_workers() const noexcept {
        return num_workers_;
    }

    // Launch a kernel function across an emulated grid of thread blocks
    void launch(Dim3 grid_dim, Dim3 block_dim, size_t shared_mem_bytes,
                const std::function<void()>& kernel_func);

    // Synchronize emulated device
    void synchronize();

    // Memory allocation
    void* allocate_device(size_t bytes);
    void free_device(void* ptr);
    void copy_h2d(void* dst, const void* src, size_t bytes);
    void copy_d2h(void* dst, const void* src, size_t bytes);
    void copy_d2d(void* dst, const void* src, size_t bytes);
    void memset_device(void* dst, int value, size_t bytes);

private:
    SimtEngine();
    ~SimtEngine();

    size_t num_workers_{4};
};

#endif // !ONYX_HAS_CUDA

} // namespace onyx::cuda
