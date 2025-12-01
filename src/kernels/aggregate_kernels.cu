#include "onyx/kernels/aggregate.cuh"

namespace onyx::kernels {

template <typename T>
__global__ void scalar_aggregate_sum_kernel(
    const T* __restrict__ data,
    const uint8_t* __restrict__ null_bitmap,
    size_t num_rows,
    double* __restrict__ out_sum,
    uint64_t* __restrict__ out_count)
{
    scalar_aggregate_sum<T>(data, null_bitmap, num_rows, out_sum, out_count);
}

template <typename KeyType, typename ValType>
__global__ void grouped_aggregate_hash_kernel(
    const KeyType* __restrict__ group_keys,
    const ValType* __restrict__ values,
    const uint8_t* __restrict__ null_bitmap,
    size_t num_rows,
    GroupAggregateSlot* __restrict__ hash_table,
    size_t table_capacity)
{
    grouped_aggregate_hash<KeyType, ValType>(group_keys, values, null_bitmap, num_rows, hash_table, table_capacity);
}

} // namespace onyx::kernels
