#pragma once

#include "onyx/cuda/warp_intrinsics.cuh"
#include "onyx/common/types.hpp"
#include <cstdint>
#include <vector>

namespace onyx::kernels {

struct SortKeyDescriptor {
    enum class Direction : uint8_t {
        Ascending = 0,
        Descending = 1
    };

    const void* column_data{nullptr};
    DataType type{TypeId::INT32};
    Direction direction{Direction::Ascending};
    bool nulls_first{false};
};

} // namespace onyx::kernels
