#include "onyx/kernels/compression.cuh"

namespace onyx::kernels {

__global__ void bitpack_unpack_u32_kernel(
    const uint64_t* __restrict__ packed_words,
    uint32_t bit_width,
    size_t num_elements,
    uint32_t* __restrict__ out_unpacked)
{
    bitpack_unpack_u32(packed_words, bit_width, num_elements, out_unpacked);
}

__global__ void frame_of_reference_decode_kernel(
    int64_t base_value,
    const int32_t* __restrict__ deltas,
    size_t num_elements,
    int64_t* __restrict__ out_values)
{
    frame_of_reference_decode(base_value, deltas, num_elements, out_values);
}

template <typename T>
__global__ void rle_decode_kernel(
    const T* __restrict__ run_values,
    const uint32_t* __restrict__ run_cumulative_lengths,
    size_t num_runs,
    size_t total_elements,
    T* __restrict__ out_uncompressed)
{
    rle_decode<T>(run_values, run_cumulative_lengths, num_runs, total_elements, out_uncompressed);
}

} // namespace onyx::kernels
