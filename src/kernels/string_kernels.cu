#include "onyx/kernels/string_ops.cuh"

namespace onyx::kernels {

__global__ void string_filter_contains_kernel(
    const char* __restrict__ char_buffer,
    const StringOffset* __restrict__ string_offsets,
    size_t num_strings,
    const char* __restrict__ pattern,
    uint32_t pattern_len,
    uint32_t* __restrict__ out_row_ids,
    uint32_t* __restrict__ out_count)
{
    string_filter_contains(char_buffer, string_offsets, num_strings, pattern, pattern_len, out_row_ids, out_count);
}

} // namespace onyx::kernels
