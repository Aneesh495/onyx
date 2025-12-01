#include "onyx/kernels/hash_join.cuh"

namespace onyx::kernels {

template <typename KeyType>
__global__ void hash_join_build_kernel(
    const KeyType* __restrict__ build_keys,
    size_t num_build_rows,
    HashBucket* __restrict__ table,
    size_t table_capacity)
{
    hash_join_build<KeyType>(build_keys, num_build_rows, table, table_capacity);
}

template <typename KeyType>
__global__ void hash_join_probe_kernel(
    const KeyType* __restrict__ probe_keys,
    size_t num_probe_rows,
    const HashBucket* __restrict__ table,
    size_t table_capacity,
    uint32_t* __restrict__ out_build_row_ids,
    uint32_t* __restrict__ out_probe_row_ids,
    uint32_t* __restrict__ out_match_count)
{
    hash_join_probe<KeyType>(probe_keys, num_probe_rows, table, table_capacity, out_build_row_ids, out_probe_row_ids, out_match_count);
}

} // namespace onyx::kernels
