#include "onyx/kernels/radix_sort.cuh"
#include "onyx/cuda/simt_engine.hpp"
#include <vector>
#include <numeric>

namespace onyx::kernels {

template <typename KeyType, typename ValueType>
void launch_radix_sort(
    KeyType* keys,
    ValueType* values,
    size_t num_elements,
    bool ascending,
    cudaStream_t stream)
{
    if (num_elements <= 1) return;

    uint32_t num_blocks = std::min<uint32_t>((num_elements + 255) / 256, 32);
    size_t hist_size = RADIX_BINS * num_blocks;

    std::vector<uint32_t> block_histograms(hist_size, 0);
    std::vector<uint32_t> global_offsets(hist_size, 0);

    std::vector<KeyType> temp_keys(num_elements);
    std::vector<ValueType> temp_values(num_elements);

    KeyType* in_k = keys;
    KeyType* out_k = temp_keys.data();
    ValueType* in_v = values;
    ValueType* out_v = temp_values.data();

    uint32_t total_bits = sizeof(KeyType) * 8;

    for (uint32_t shift = 0; shift < total_bits; shift += RADIX_BITS) {
        // Step 1: Compute histograms
        std::fill(block_histograms.begin(), block_histograms.end(), 0);

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        // Launch native CUDA kernel
#else
        cuda::SimtEngine::instance().launch(
            cuda::Dim3(num_blocks), cuda::Dim3(256), 0,
            [&]() {
                radix_sort_histogram<KeyType>(in_k, num_elements, shift, block_histograms.data(), num_blocks);
            });
#endif

        // Step 2: Global prefix scan across bins and blocks
        uint32_t running_sum = 0;
        for (uint32_t bin = 0; bin < RADIX_BINS; ++bin) {
            for (uint32_t b = 0; b < num_blocks; ++b) {
                size_t idx = bin * num_blocks + b;
                global_offsets[idx] = running_sum;
                running_sum += block_histograms[idx];
            }
        }

        // Step 3: Scatter to output buffer
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
        // Launch native CUDA scatter
#else
        cuda::SimtEngine::instance().launch(
            cuda::Dim3(num_blocks), cuda::Dim3(256), 0,
            [&]() {
                radix_sort_scatter<KeyType, ValueType>(
                    in_k, in_v, out_k, out_v, num_elements, shift, global_offsets.data(), num_blocks);
            });
#endif

        // Swap ping-pong buffers
        std::swap(in_k, out_k);
        std::swap(in_v, out_v);
    }

    // If result ended up in temp buffer, copy back
    if (in_k != keys) {
        std::memcpy(keys, in_k, num_elements * sizeof(KeyType));
        if (values) {
            std::memcpy(values, in_v, num_elements * sizeof(ValueType));
        }
    }

    // Reverse if descending
    if (!ascending) {
        for (size_t i = 0; i < num_elements / 2; ++i) {
            std::swap(keys[i], keys[num_elements - 1 - i]);
            if (values) {
                std::swap(values[i], values[num_elements - 1 - i]);
            }
        }
    }
}

// Instantiations
template void launch_radix_sort<int32_t, uint32_t>(int32_t*, uint32_t*, size_t, bool, cudaStream_t);
template void launch_radix_sort<uint32_t, uint32_t>(uint32_t*, uint32_t*, size_t, bool, cudaStream_t);
template void launch_radix_sort<int64_t, uint32_t>(int64_t*, uint32_t*, size_t, bool, cudaStream_t);
template void launch_radix_sort<uint64_t, uint32_t>(uint64_t*, uint32_t*, size_t, bool, cudaStream_t);

} // namespace onyx::kernels
