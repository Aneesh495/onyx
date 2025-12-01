#include "onyx/kernels/hash_join.cuh"
#include "onyx/cuda/simt_engine.hpp"
#include "onyx/common/bit_utils.hpp"
#include <vector>

namespace onyx::kernels {

template <typename KeyType>
void launch_hash_join(
    const KeyType* build_keys,
    size_t num_build_rows,
    const KeyType* probe_keys,
    size_t num_probe_rows,
    uint32_t* out_build_row_ids,
    uint32_t* out_probe_row_ids,
    uint32_t* out_match_count,
    cudaStream_t stream)
{
    if (num_build_rows == 0 || num_probe_rows == 0) {
        *out_match_count = 0;
        return;
    }

    // Allocate hash table on device (capacity is power of 2, load factor ~0.5)
    size_t table_capacity = bit_utils::next_power_of_two(num_build_rows * 2);
    std::vector<HashBucket> table(table_capacity);
    for (auto& b : table) {
        b.key = HASH_EMPTY_KEY;
        b.row_id = 0;
        b.padding = 0;
    }

    uint32_t num_blocks = std::min<uint32_t>((num_build_rows + 255) / 256, 64);

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA launch
#else
    // Phase 1: Build
    cuda::SimtEngine::instance().launch(
        cuda::Dim3(num_blocks), cuda::Dim3(256), 0,
        [&]() {
            hash_join_build<KeyType>(build_keys, num_build_rows, table.data(), table_capacity);
        });

    // Phase 2: Probe
    uint32_t probe_blocks = std::min<uint32_t>((num_probe_rows + 255) / 256, 64);
    cuda::SimtEngine::instance().launch(
        cuda::Dim3(probe_blocks), cuda::Dim3(256), 0,
        [&]() {
            hash_join_probe<KeyType>(
                probe_keys, num_probe_rows, table.data(), table_capacity,
                out_build_row_ids, out_probe_row_ids, out_match_count);
        });
#endif
}

// Instantiations
template void launch_hash_join<int32_t>(const int32_t*, size_t, const int32_t*, size_t, uint32_t*, uint32_t*, uint32_t*, cudaStream_t);
template void launch_hash_join<uint32_t>(const uint32_t*, size_t, const uint32_t*, size_t, uint32_t*, uint32_t*, uint32_t*, cudaStream_t);
template void launch_hash_join<int64_t>(const int64_t*, size_t, const int64_t*, size_t, uint32_t*, uint32_t*, uint32_t*, cudaStream_t);
template void launch_hash_join<uint64_t>(const uint64_t*, size_t, const uint64_t*, size_t, uint32_t*, uint32_t*, uint32_t*, cudaStream_t);

} // namespace onyx::kernels
