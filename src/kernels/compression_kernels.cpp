#include "onyx/kernels/compression.cuh"
#include "onyx/cuda/simt_engine.hpp"

namespace onyx::kernels {

void launch_bitpack_unpack(
    const uint64_t* packed_words,
    uint32_t bit_width,
    size_t num_elements,
    uint32_t* out_unpacked,
    cudaStream_t stream)
{
    if (num_elements == 0) return;
    uint32_t num_blocks = std::min<uint32_t>((num_elements + 255) / 256, 64);

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA
#else
    cuda::SimtEngine::instance().launch(
        cuda::Dim3(num_blocks), cuda::Dim3(256), 0,
        [&]() {
            bitpack_unpack_u32(packed_words, bit_width, num_elements, out_unpacked);
        });
#endif
}

void launch_for_decode(
    int64_t base_value,
    const int32_t* deltas,
    size_t num_elements,
    int64_t* out_values,
    cudaStream_t stream)
{
    if (num_elements == 0) return;
    uint32_t num_blocks = std::min<uint32_t>((num_elements + 255) / 256, 64);

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA
#else
    cuda::SimtEngine::instance().launch(
        cuda::Dim3(num_blocks), cuda::Dim3(256), 0,
        [&]() {
            frame_of_reference_decode(base_value, deltas, num_elements, out_values);
        });
#endif
}

template <typename T>
void launch_rle_decode(
    const T* run_values,
    const uint32_t* run_cumulative_lengths,
    size_t num_runs,
    size_t total_elements,
    T* out_uncompressed,
    cudaStream_t stream)
{
    if (total_elements == 0) return;
    uint32_t num_blocks = std::min<uint32_t>((total_elements + 255) / 256, 64);

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA
#else
    cuda::SimtEngine::instance().launch(
        cuda::Dim3(num_blocks), cuda::Dim3(256), 0,
        [&]() {
            rle_decode<T>(run_values, run_cumulative_lengths, num_runs, total_elements, out_uncompressed);
        });
#endif
}

// Instantiations
template void launch_rle_decode<int32_t>(const int32_t*, const uint32_t*, size_t, size_t, int32_t*, cudaStream_t);
template void launch_rle_decode<int64_t>(const int64_t*, const uint32_t*, size_t, size_t, int64_t*, cudaStream_t);
template void launch_rle_decode<uint32_t>(const uint32_t*, const uint32_t*, size_t, size_t, uint32_t*, cudaStream_t);

} // namespace onyx::kernels
