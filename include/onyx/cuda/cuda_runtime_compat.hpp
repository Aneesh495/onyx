#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <stdexcept>

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
#include <cuda_runtime.h>
#include <cuda.h>

#define ONYX_HOST_DEVICE __host__ __device__
#define ONYX_DEVICE __device__
#define ONYX_HOST __host__
#define ONYX_GLOBAL __global__
#define ONYX_SHARED __shared__
#define ONYX_CONSTANT __constant__
#define ONYX_INLINE __forceinline__

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            throw ::onyx::CudaKernelException(                                 \
                std::string("CUDA Error: ") + cudaGetErrorString(_err) +       \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));           \
        }                                                                      \
    } while (0)

#else // SIMT CPU Emulation mode

#define ONYX_HOST_DEVICE inline
#define ONYX_DEVICE inline
#define ONYX_HOST inline
#define ONYX_GLOBAL inline
#define ONYX_SHARED static thread_local
#define ONYX_CONSTANT const
#define ONYX_INLINE

enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 1,
    cudaErrorInitializationError = 2,
    cudaErrorLaunchFailure = 3,
    cudaErrorInvalidValue = 4,
    cudaErrorInvalidDevicePointer = 5,
    cudaErrorUnknown = 999
};

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

using cudaStream_t = void*;
using cudaEvent_t = void*;
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;

inline const char* cudaGetErrorString(cudaError_t err) {
    switch (err) {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
        case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
        case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
        default: return "cudaErrorUnknown";
    }
}

// Forward declare SIMT engine memory methods
namespace onyx::cuda {
void* simt_alloc_device(size_t bytes);
void simt_free_device(void* ptr);
void simt_copy_h2d(void* dst, const void* src, size_t bytes);
void simt_copy_d2h(void* dst, const void* src, size_t bytes);
void simt_copy_d2d(void* dst, const void* src, size_t bytes);
void simt_memset(void* dst, int value, size_t bytes);
void simt_sync();
}

inline cudaError_t cudaMalloc(void** dev_ptr, size_t size) {
    if (!dev_ptr) return cudaErrorInvalidValue;
    *dev_ptr = ::onyx::cuda::simt_alloc_device(size);
    return *dev_ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

inline cudaError_t cudaFree(void* dev_ptr) {
    ::onyx::cuda::simt_free_device(dev_ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    if (!dst || !src) return count == 0 ? cudaSuccess : cudaErrorInvalidValue;
    switch (kind) {
        case cudaMemcpyHostToDevice:
            ::onyx::cuda::simt_copy_h2d(dst, src, count);
            break;
        case cudaMemcpyDeviceToHost:
            ::onyx::cuda::simt_copy_d2h(dst, src, count);
            break;
        case cudaMemcpyDeviceToDevice:
            ::onyx::cuda::simt_copy_d2d(dst, src, count);
            break;
        case cudaMemcpyHostToHost:
        case cudaMemcpyDefault:
            std::memcpy(dst, src, count);
            break;
    }
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, [[maybe_unused]] cudaStream_t stream = nullptr) {
    return cudaMemcpy(dst, src, count, kind);
}

inline cudaError_t cudaMemset(void* dev_ptr, int value, size_t count) {
    if (!dev_ptr) return count == 0 ? cudaSuccess : cudaErrorInvalidValue;
    ::onyx::cuda::simt_memset(dev_ptr, value, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemsetAsync(void* dev_ptr, int value, size_t count, [[maybe_unused]] cudaStream_t stream = nullptr) {
    return cudaMemset(dev_ptr, value, count);
}

inline cudaError_t cudaDeviceSynchronize() {
    ::onyx::cuda::simt_sync();
    return cudaSuccess;
}

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            throw ::onyx::CudaKernelException(                                 \
                std::string("Emulated CUDA Error: ") +                         \
                cudaGetErrorString(_err) + " at " + __FILE__ + ":" +           \
                std::to_string(__LINE__));                                     \
        }                                                                      \
    } while (0)

#endif // ONYX_HAS_CUDA
