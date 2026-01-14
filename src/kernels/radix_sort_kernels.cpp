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

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
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

        // Step 2: Global prefix scan across bins and blocks
        uint32_t running_sum = 0;
        for (uint32_t bin = 0; bin < RADIX_BINS; ++bin) {
            for (uint32_t b = 0; b < num_blocks; ++b) {
                size_t idx = bin * num_blocks + b;
                global_offsets[idx] = running_sum;
                running_sum += block_histograms[idx];
            }
        }

        // Swap ping-pong buffers
        std::swap(in_k, out_k);
        std::swap(in_v, out_v);
    }

    if (in_k != keys) {
        std::memcpy(keys, in_k, num_elements * sizeof(KeyType));
        if (values) {
            std::memcpy(values, in_v, num_elements * sizeof(ValueType));
        }
    }
#else
    std::vector<size_t> indices(num_elements);
    std::iota(indices.begin(), indices.end(), 0);
    if (ascending) {
        std::stable_sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return keys[a] < keys[b];
        });
    } else {
        std::stable_sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return keys[a] > keys[b];
        });
    }
    std::vector<KeyType> sorted_keys(num_elements);
    std::vector<ValueType> sorted_values(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        sorted_keys[i] = keys[indices[i]];
        if (values) sorted_values[i] = values[indices[i]];
    }
    std::memcpy(keys, sorted_keys.data(), num_elements * sizeof(KeyType));
    if (values) std::memcpy(values, sorted_values.data(), num_elements * sizeof(ValueType));
#endif
}

// Instantiations
template void launch_radix_sort<int32_t, uint32_t>(int32_t*, uint32_t*, size_t, bool, cudaStream_t);
template void launch_radix_sort<uint32_t, uint32_t>(uint32_t*, uint32_t*, size_t, bool, cudaStream_t);
template void launch_radix_sort<int64_t, uint32_t>(int64_t*, uint32_t*, size_t, bool, cudaStream_t);
template void launch_radix_sort<uint64_t, uint32_t>(uint64_t*, uint32_t*, size_t, bool, cudaStream_t);

} // namespace onyx::kernels
