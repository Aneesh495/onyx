#include "test_framework.hpp"
#include "onyx/cuda/warp_intrinsics.cuh"
#include "onyx/cuda/cooperative_primitives.cuh"
#include <iostream>
#include <cassert>


ONYX_TEST(WarpVotingAndActiveMask) {
    // Test warp ballot emulation
    uint32_t b1 = onyx::cuda::warp_ballot(1);
    ONYX_ASSERT_EQ(b1, 0xFFFFFFFFU);

    uint32_t b2 = onyx::cuda::warp_ballot(0);
    ONYX_ASSERT_EQ(b2, 0U);
}

ONYX_TEST(WarpShuffleReductions) {
    // Test warp reduction sum
    int val = 1;
    int sum = onyx::cuda::warp_reduce_sum(val);
    ONYX_ASSERT_EQ(sum, 32);

    int max_val = onyx::cuda::warp_reduce_max(42);
    ONYX_ASSERT_EQ(max_val, 42);

    int min_val = onyx::cuda::warp_reduce_min(17);
    ONYX_ASSERT_EQ(min_val, 17);
}

ONYX_TEST(WarpPrefixScan) {
    // Kogge-Stone inclusive scan
    int val = 1;
    int inclusive = onyx::cuda::warp_scan_inclusive(val);
    // In warp lane 0, inclusive is 1
    ONYX_ASSERT(inclusive >= 1);
}
