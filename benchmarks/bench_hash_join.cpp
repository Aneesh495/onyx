#include "onyx/kernels/hash_join.cuh"
#include <vector>
#include <functional>
#include <string>

namespace onyx::kernels {
template <typename KeyType>
void launch_hash_join(
    const KeyType* build_keys, size_t num_build_rows,
    const KeyType* probe_keys, size_t num_probe_rows,
    uint32_t* out_build_row_ids, uint32_t* out_probe_row_ids,
    uint32_t* out_match_count, cudaStream_t stream);
}

extern void register_benchmark(std::string name, size_t num_elements, std::function<void()> func);

#define ONYX_BENCH(name, num_elements) \
    static void bench_##name(); \
    static int dummy_##name = (register_benchmark(#name, num_elements, bench_##name), 0); \
    static void bench_##name()

ONYX_BENCH(HashJoin_OpenAddressing, 500000) {
    static std::vector<int32_t> build(500000);
    static std::vector<int32_t> probe(500000);
    static std::vector<uint32_t> out_b(500000);
    static std::vector<uint32_t> out_p(500000);
    static bool init = false;
    if (!init) {
        for (size_t i = 0; i < 500000; ++i) {
            build[i] = static_cast<int32_t>(i * 2);
            probe[i] = static_cast<int32_t>(i);
        }
        init = true;
    }

    uint32_t match_count = 0;
    onyx::kernels::launch_hash_join<int32_t>(
        build.data(), build.size(),
        probe.data(), probe.size(),
        out_b.data(), out_p.data(),
        &match_count, nullptr);
}
