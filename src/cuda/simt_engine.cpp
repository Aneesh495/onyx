#include "onyx/cuda/simt_engine.hpp"
#include "onyx/cuda/warp_intrinsics.cuh"
#include <cstdlib>
#include <thread>
#include <mutex>
#include <vector>

namespace onyx::cuda {

#if !defined(ONYX_HAS_CUDA) || ONYX_HAS_CUDA == 0

thread_local SimtWarpContext g_simt_warp_ctx;
thread_local SimtThreadContext g_simt_thread_ctx;

SimtEngine::SimtEngine() {
    unsigned int hardware_threads = std::thread::hardware_concurrency();
    num_workers_ = hardware_threads > 0 ? hardware_threads : 4;
}

SimtEngine::~SimtEngine() = default;

void* SimtEngine::allocate_device(size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    int res = posix_memalign(&ptr, 128, bytes);
    if (res != 0) return nullptr;
    return ptr;
}

void SimtEngine::free_device(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

void SimtEngine::copy_h2d(void* dst, const void* src, size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void SimtEngine::copy_d2h(void* dst, const void* src, size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void SimtEngine::copy_d2d(void* dst, const void* src, size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void SimtEngine::memset_device(void* dst, int value, size_t bytes) {
    std::memset(dst, value, bytes);
}

void SimtEngine::synchronize() {
}

void SimtEngine::launch(Dim3 grid_dim, Dim3 block_dim, [[maybe_unused]] size_t shared_mem_bytes,
                        const std::function<void()>& kernel_func) {
    const uint32_t threads_per_block = block_dim.x * block_dim.y * block_dim.z;
    const uint32_t warps_per_block = (threads_per_block + 31) / 32;

    for (uint32_t bz = 0; bz < grid_dim.z; ++bz) {
        for (uint32_t by = 0; by < grid_dim.y; ++by) {
            for (uint32_t bx = 0; bx < grid_dim.x; ++bx) {
                g_simt_thread_ctx.grid_dim = grid_dim;
                g_simt_thread_ctx.block_dim = block_dim;
                g_simt_thread_ctx.block_idx = Dim3(bx, by, bz);

                for (uint32_t wid = 0; wid < warps_per_block; ++wid) {
                    g_simt_warp_ctx.warp_idx = wid;

                    for (uint32_t lid = 0; lid < 32; ++lid) {
                        uint32_t tid = wid * 32 + lid;
                        if (tid >= threads_per_block) continue;

                        uint32_t tz = tid / (block_dim.x * block_dim.y);
                        uint32_t rem = tid % (block_dim.x * block_dim.y);
                        uint32_t ty = rem / block_dim.x;
                        uint32_t tx = rem % block_dim.x;

                        g_simt_thread_ctx.thread_idx = Dim3(tx, ty, tz);
                        g_simt_warp_ctx.lane_idx = lid;

                        kernel_func();
                    }
                }
            }
        }
    }
}

void* simt_alloc_device(size_t bytes) {
    return SimtEngine::instance().allocate_device(bytes);
}

void simt_free_device(void* ptr) {
    SimtEngine::instance().free_device(ptr);
}

void simt_copy_h2d(void* dst, const void* src, size_t bytes) {
    SimtEngine::instance().copy_h2d(dst, src, bytes);
}

void simt_copy_d2h(void* dst, const void* src, size_t bytes) {
    SimtEngine::instance().copy_d2h(dst, src, bytes);
}

void simt_copy_d2d(void* dst, const void* src, size_t bytes) {
    SimtEngine::instance().copy_d2d(dst, src, bytes);
}

void simt_memset(void* dst, int value, size_t bytes) {
    SimtEngine::instance().memset_device(dst, value, bytes);
}

void simt_sync() {
    SimtEngine::instance().synchronize();
}

#endif // !ONYX_HAS_CUDA

} // namespace onyx::cuda
