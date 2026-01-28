#pragma once

#include "onyx/cuda/cuda_runtime_compat.hpp"
#include "onyx/kernels/filter.cuh"
#include "onyx/kernels/aggregate.cuh"
#include "onyx/kernels/string_ops.cuh"
#include <cstdint>
#include <vector>

namespace onyx::kernels {

template <typename T>
void launch_filter_column(
    const T* col_data,
    const uint8_t* null_bitmap,
    size_t num_rows,
    T literal,
    CompareOp op,
    uint32_t* out_row_ids,
    uint32_t* out_count,
    cudaStream_t stream = nullptr);

template <typename KeyType, typename ValueType>
void launch_radix_sort(
    KeyType* keys,
    ValueType* values,
    size_t num_elements,
    bool ascending = true,
    cudaStream_t stream = nullptr);

template <typename KeyType>
void launch_hash_join(
    const KeyType* build_keys,
    size_t num_build_rows,
    const KeyType* probe_keys,
    size_t num_probe_rows,
    uint32_t* out_build_row_ids,
    uint32_t* out_probe_row_ids,
    uint32_t* out_match_count,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_scalar_aggregate(
    const T* data,
    const uint8_t* null_bitmap,
    size_t num_rows,
    double* out_sum,
    uint64_t* out_count,
    cudaStream_t stream = nullptr);

template <typename KeyType, typename ValType>
void launch_grouped_aggregate(
    const KeyType* group_keys,
    const ValType* values,
    const uint8_t* null_bitmap,
    size_t num_rows,
    std::vector<GroupAggregateSlot>& out_table,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_cumulative_sum(
    const T* in_data,
    size_t num_rows,
    double* out_cumsum,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_sliding_average(
    const T* in_data,
    size_t num_rows,
    uint32_t window_size,
    double* out_avg,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_lag(
    const T* in_data,
    size_t num_rows,
    uint32_t offset,
    T default_val,
    T* out_data,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_lead(
    const T* in_data,
    size_t num_rows,
    uint32_t offset,
    T default_val,
    T* out_data,
    cudaStream_t stream = nullptr);

void launch_bitpack_unpack(
    const uint64_t* packed_words,
    uint32_t bit_width,
    size_t num_elements,
    uint32_t* out_unpacked,
    cudaStream_t stream = nullptr);

void launch_for_decode(
    int64_t base_value,
    const int32_t* deltas,
    size_t num_elements,
    int64_t* out_values,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_rle_decode(
    const T* run_values,
    const uint32_t* run_cumulative_lengths,
    size_t num_runs,
    size_t total_elements,
    T* out_uncompressed,
    cudaStream_t stream = nullptr);

void launch_string_filter_contains(
    const char* char_buffer,
    const StringOffset* string_offsets,
    size_t num_strings,
    const char* pattern,
    uint32_t pattern_len,
    uint32_t* out_row_ids,
    uint32_t* out_count,
    cudaStream_t stream = nullptr);

template <typename KeyType, typename ValType>
void launch_fused_filter_project_sum(
    const KeyType* filter_col,
    KeyType filter_literal,
    CompareOp filter_op,
    const ValType* col_a,
    const ValType* col_b,
    size_t num_rows,
    double* out_sum,
    uint64_t* out_count,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_dictionary_decode(
    const uint32_t* encoded_codes,
    const T* dictionary_entries,
    size_t num_dict_entries,
    size_t num_rows,
    T* out_decoded,
    cudaStream_t stream = nullptr);

} // namespace onyx::kernels
