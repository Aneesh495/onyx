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

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA launch
    size_t table_capacity = bit_utils::next_power_of_two(num_build_rows * 2);
    std::vector<HashBucket> table(table_capacity);
    for (auto& b : table) {
        b.key = HASH_EMPTY_KEY;
        b.row_id = 0;
        b.padding = 0;
    }

    uint32_t num_blocks = std::min<uint32_t>((num_build_rows + 255) / 256, 64);
    uint32_t probe_blocks = std::min<uint32_t>((num_probe_rows + 255) / 256, 64);
#else
    uint32_t match_count = 0;
    std::unordered_multimap<KeyType, uint32_t> map;
    for (size_t b = 0; b < num_build_rows; ++b) {
        map.insert({build_keys[b], static_cast<uint32_t>(b)});
    }
    for (size_t p = 0; p < num_probe_rows; ++p) {
        auto range = map.equal_range(probe_keys[p]);
        for (auto it = range.first; it != range.second; ++it) {
            out_build_row_ids[match_count] = it->second;
            out_probe_row_ids[match_count] = static_cast<uint32_t>(p);
            match_count++;
        }
    }
    *out_match_count = match_count;
#endif
}

// Instantiations
template void launch_hash_join<int32_t>(const int32_t*, size_t, const int32_t*, size_t, uint32_t*, uint32_t*, uint32_t*, cudaStream_t);
template void launch_hash_join<uint32_t>(const uint32_t*, size_t, const uint32_t*, size_t, uint32_t*, uint32_t*, uint32_t*, cudaStream_t);
template void launch_hash_join<int64_t>(const int64_t*, size_t, const int64_t*, size_t, uint32_t*, uint32_t*, uint32_t*, cudaStream_t);
template void launch_hash_join<uint64_t>(const uint64_t*, size_t, const uint64_t*, size_t, uint32_t*, uint32_t*, uint32_t*, cudaStream_t);

} // namespace onyx::kernels
