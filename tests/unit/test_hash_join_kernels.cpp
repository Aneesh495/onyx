#include "onyx/kernels/hash_join.cuh"
#include <vector>
#include <stdexcept>

namespace onyx::kernels {
template <typename KeyType>
void launch_hash_join(
    const KeyType* build_keys, size_t num_build_rows,
    const KeyType* probe_keys, size_t num_probe_rows,
    uint32_t* out_build_row_ids, uint32_t* out_probe_row_ids,
    uint32_t* out_match_count, cudaStream_t stream);
}

extern void register_test(std::string name, std::function<void()> func);

#define ONYX_TEST(name) \
    static void test_##name(); \
    static int dummy_##name = (register_test(#name, test_##name), 0); \
    static void test_##name()

#define ONYX_ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while (0)
#define ONYX_ASSERT_EQ(a, b) do { if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b); } while (0)

ONYX_TEST(HashJoinBasicMatching) {
    std::vector<int32_t> build_keys = {101, 102, 103, 104};
    std::vector<int32_t> probe_keys = {103, 101, 999, 104, 888};

    std::vector<uint32_t> out_build(probe_keys.size());
    std::vector<uint32_t> out_probe(probe_keys.size());
    uint32_t match_count = 0;

    onyx::kernels::launch_hash_join<int32_t>(
        build_keys.data(), build_keys.size(),
        probe_keys.data(), probe_keys.size(),
        out_build.data(), out_probe.data(),
        &match_count, nullptr);

    // 103, 101, 104 match -> 3 matches
    ONYX_ASSERT_EQ(match_count, 3);
}
