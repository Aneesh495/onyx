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
#define ONYX_INLINE inline

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
