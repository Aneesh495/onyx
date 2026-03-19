#include "onyx/cuda/warp_intrinsics.cuh"
#include "onyx/cuda/cooperative_primitives.cuh"
#include <iostream>
#include <vector>
#include <functional>
#include <string>

extern void register_benchmark(std::string name, size_t num_elements, std::function<void()> func);

static void bench_warp_shuffle_sum() {
    constexpr size_t N = 1000000;
    volatile int total = 0;
    for (size_t i = 0; i < N; ++i) {
        total += onyx::cuda::warp_reduce_sum(static_cast<int>(i & 31));
    }
}

static void bench_warp_prefix_scan() {
    constexpr size_t N = 1000000;
    volatile int total = 0;
    for (size_t i = 0; i < N; ++i) {
        total += onyx::cuda::warp_scan_inclusive(static_cast<int>(i & 31));
    }
}

int reg_warp_bench_1 = (register_benchmark("Warp_Intrinsics_ShuffleSum", 1000000, bench_warp_shuffle_sum), 0);
int reg_warp_bench_2 = (register_benchmark("Warp_Intrinsics_PrefixScan", 1000000, bench_warp_prefix_scan), 0);
