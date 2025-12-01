#include "onyx/kernels/aggregate.cuh"
#include "onyx/cuda/simt_engine.hpp"
#include "onyx/common/bit_utils.hpp"
#include <vector>

namespace onyx::kernels {

template <typename T>
void launch_scalar_aggregate(
    const T* data,
    const uint8_t* null_bitmap,
    size_t num_rows,
    double* out_sum,
    uint64_t* out_count,
    cudaStream_t stream)
{
    if (num_rows == 0) {
        *out_sum = 0.0;
        *out_count = 0;
        return;
    }

    uint32_t num_blocks = std::min<uint32_t>((num_rows + 255) / 256, 64);

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA launch
#else
    cuda::SimtEngine::instance().launch(
        cuda::Dim3(num_blocks), cuda::Dim3(256), 0,
        [&]() {
            scalar_aggregate_sum<T>(data, null_bitmap, num_rows, out_sum, out_count);
        });
#endif
}

template <typename KeyType, typename ValType>
void launch_grouped_aggregate(
    const KeyType* group_keys,
    const ValType* values,
    const uint8_t* null_bitmap,
    size_t num_rows,
    std::vector<GroupAggregateSlot>& out_table,
    cudaStream_t stream)
{
    if (num_rows == 0) return;

    size_t table_capacity = bit_utils::next_power_of_two(std::max(num_rows * 2, static_cast<size_t>(1024)));
    out_table.resize(table_capacity);
    for (auto& slot : out_table) {
        slot.group_key = 0xFFFFFFFF80000000ULL;
        slot.sum = 0.0;
        slot.count = 0;
        slot.min_val = 1e30;
        slot.max_val = -1e30;
    }

    uint32_t num_blocks = std::min<uint32_t>((num_rows + 255) / 256, 64);

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    // Native CUDA launch
#else
    cuda::SimtEngine::instance().launch(
        cuda::Dim3(num_blocks), cuda::Dim3(256), 0,
        [&]() {
            grouped_aggregate_hash<KeyType, ValType>(
                group_keys, values, null_bitmap, num_rows, out_table.data(), table_capacity);
        });
#endif
}

// Instantiations
template void launch_scalar_aggregate<int32_t>(const int32_t*, const uint8_t*, size_t, double*, uint64_t*, cudaStream_t);
template void launch_scalar_aggregate<int64_t>(const int64_t*, const uint8_t*, size_t, double*, uint64_t*, cudaStream_t);
template void launch_scalar_aggregate<float>(const float*, const uint8_t*, size_t, double*, uint64_t*, cudaStream_t);
template void launch_scalar_aggregate<double>(const double*, const uint8_t*, size_t, double*, uint64_t*, cudaStream_t);

template void launch_grouped_aggregate<int32_t, int32_t>(const int32_t*, const int32_t*, const uint8_t*, size_t, std::vector<GroupAggregateSlot>&, cudaStream_t);
template void launch_grouped_aggregate<int64_t, double>(const int64_t*, const double*, const uint8_t*, size_t, std::vector<GroupAggregateSlot>&, cudaStream_t);
template void launch_grouped_aggregate<uint32_t, float>(const uint32_t*, const float*, const uint8_t*, size_t, std::vector<GroupAggregateSlot>&, cudaStream_t);

} // namespace onyx::kernels
