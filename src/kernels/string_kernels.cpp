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
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA
    uint32_t num_blocks = std::min<uint32_t>((num_strings + 255) / 256, 64);
#else
    std::string_view pat(pattern, pattern_len);
    uint32_t count = 0;
    for (size_t i = 0; i < num_strings; ++i) {
        uint32_t off = string_offsets[i].offset;
        uint32_t len = string_offsets[i].length;
        std::string_view s(char_buffer + off, len);
        if (s.find(pat) != std::string_view::npos) {
            out_row_ids[count++] = static_cast<uint32_t>(i);
        }
    }
    *out_count = count;
#endif
}

} // namespace onyx::kernels
