#include "test_framework.hpp"
#include "onyx/kernels/filter.cuh"
#include "onyx/kernels/aggregate.cuh"
#include "onyx/kernels/radix_sort.cuh"
#include <vector>
#include <random>
#include <stdexcept>
#include <algorithm>

namespace onyx::kernels {
template <typename T>
void launch_filter_column(
    const T* col_data, const uint8_t* null_bitmap, size_t num_rows,
    T literal, CompareOp op, uint32_t* out_row_ids, uint32_t* out_count, cudaStream_t stream);

template <typename T>
void launch_scalar_aggregate(
    const T* data, const uint8_t* null_bitmap, size_t num_rows,
    double* out_sum, uint64_t* out_count, cudaStream_t stream);

template <typename KeyType, typename ValueType>
void launch_radix_sort(KeyType* keys, ValueType* values, size_t num_elements, bool ascending, cudaStream_t stream);
}


ONYX_TEST(BitForBitFilterEquivalence) {
    constexpr size_t N = 10000;
    std::mt19937 rng(1337);
    std::uniform_int_distribution<int32_t> dist(-5000, 5000);

    std::vector<int32_t> data(N);
    for (size_t i = 0; i < N; ++i) data[i] = dist(rng);

    // Reference CPU scalar filter
    std::vector<uint32_t> ref_matches;
    for (size_t i = 0; i < N; ++i) {
        if (data[i] > 1000) ref_matches.push_back(static_cast<uint32_t>(i));
    }

    // GPU kernel filter
    std::vector<uint32_t> gpu_matches(N);
    uint32_t gpu_count = 0;
    onyx::kernels::launch_filter_column<int32_t>(
        data.data(), nullptr, N, 1000, onyx::kernels::CompareOp::GreaterThan,
        gpu_matches.data(), &gpu_count, nullptr);

    ONYX_ASSERT_EQ(gpu_count, ref_matches.size());
    gpu_matches.resize(gpu_count);
    std::sort(gpu_matches.begin(), gpu_matches.end());
    ONYX_ASSERT(gpu_matches == ref_matches);
}

ONYX_TEST(BitForBitSortEquivalence) {
    constexpr size_t N = 5000;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, 100000);

    std::vector<int32_t> data(N);
    for (size_t i = 0; i < N; ++i) data[i] = dist(rng);

    auto ref_data = data;
    std::sort(ref_data.begin(), ref_data.end());

    std::vector<uint32_t> payloads(N);
    for (size_t i = 0; i < N; ++i) payloads[i] = static_cast<uint32_t>(i);

    onyx::kernels::launch_radix_sort<int32_t, uint32_t>(
        data.data(), payloads.data(), N, true, nullptr);

    ONYX_ASSERT(data == ref_data);
}
