#include "onyx/kernels/string_ops.cuh"
#include "onyx/cuda/simt_engine.hpp"

namespace onyx::kernels {

void launch_string_filter_contains(
    const char* char_buffer,
    const StringOffset* string_offsets,
    size_t num_strings,
    const char* pattern,
    uint32_t pattern_len,
    uint32_t* out_row_ids,
    uint32_t* out_count,
    cudaStream_t stream)
{
    if (num_strings == 0) return;
    uint32_t num_blocks = std::min<uint32_t>((num_strings + 255) / 256, 64);

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA
#else
    cuda::SimtEngine::instance().launch(
        cuda::Dim3(num_blocks), cuda::Dim3(256), 0,
        [&]() {
            string_filter_contains(
                char_buffer, string_offsets, num_strings, pattern, pattern_len, out_row_ids, out_count);
        });
#endif
}

} // namespace onyx::kernels
