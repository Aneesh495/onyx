#include "onyx/kernels/multi_radix_sort.cuh"
#include "onyx/kernels/kernel_launchers.hpp"
#include <numeric>
#include <algorithm>

namespace onyx::kernels {

void launch_multi_column_sort(
    const std::vector<SortKeyDescriptor>& keys,
    size_t num_rows,
    uint32_t* out_row_ids,
    cudaStream_t stream)
{
    if (num_rows == 0) return;

    std::iota(out_row_ids, out_row_ids + num_rows, 0);
    if (keys.empty() || num_rows <= 1) return;

    // Stable LSD sorting from least-significant key to most-significant key
    for (int k = static_cast<int>(keys.size()) - 1; k >= 0; --k) {
        const auto& desc = keys[k];
        bool asc = (desc.direction == SortKeyDescriptor::Direction::Ascending);

        if (desc.type.id == TypeId::INT32) {
            const int32_t* data = static_cast<const int32_t*>(desc.column_data);
            std::stable_sort(out_row_ids, out_row_ids + num_rows, [data, asc](uint32_t a, uint32_t b) {
                return asc ? (data[a] < data[b]) : (data[a] > data[b]);
            });
        } else if (desc.type.id == TypeId::INT64) {
            const int64_t* data = static_cast<const int64_t*>(desc.column_data);
            std::stable_sort(out_row_ids, out_row_ids + num_rows, [data, asc](uint32_t a, uint32_t b) {
                return asc ? (data[a] < data[b]) : (data[a] > data[b]);
            });
        } else if (desc.type.id == TypeId::FLOAT64) {
            const double* data = static_cast<const double*>(desc.column_data);
            std::stable_sort(out_row_ids, out_row_ids + num_rows, [data, asc](uint32_t a, uint32_t b) {
                return asc ? (data[a] < data[b]) : (data[a] > data[b]);
            });
        }
    }
}

} // namespace onyx::kernels
