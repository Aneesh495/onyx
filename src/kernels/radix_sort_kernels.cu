#include "onyx/kernels/radix_sort.cuh"

namespace onyx::kernels {

template <typename KeyType>
__global__ void radix_sort_histogram_kernel(
    const KeyType* __restrict__ keys,
    size_t num_elements,
    uint32_t bit_shift,
    uint32_t* __restrict__ block_histograms,
    uint32_t num_blocks)
{
    radix_sort_histogram<KeyType>(keys, num_elements, bit_shift, block_histograms, num_blocks);
}

template <typename KeyType, typename ValueType>
__global__ void radix_sort_scatter_kernel(
    const KeyType* __restrict__ in_keys,
    const ValueType* __restrict__ in_values,
    KeyType* __restrict__ out_keys,
    ValueType* __restrict__ out_values,
    size_t num_elements,
    uint32_t bit_shift,
    const uint32_t* __restrict__ global_offsets,
    uint32_t num_blocks)
{
    radix_sort_scatter<KeyType, ValueType>(in_keys, in_values, out_keys, out_values, num_elements, bit_shift, global_offsets, num_blocks);
}

} // namespace onyx::kernels
