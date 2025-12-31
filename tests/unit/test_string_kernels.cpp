#include "onyx/kernels/string_ops.cuh"
#include <vector>
#include <string>
#include <stdexcept>

namespace onyx::kernels {
void launch_string_filter_contains(
    const char* char_buffer, const StringOffset* string_offsets, size_t num_strings,
    const char* pattern, uint32_t pattern_len, uint32_t* out_row_ids, uint32_t* out_count, cudaStream_t stream);
}

extern void register_test(std::string name, std::function<void()> func);

#define ONYX_TEST(name) \
    static void test_##name(); \
    static int dummy_##name = (register_test(#name, test_##name), 0); \
    static void test_##name()

#define ONYX_ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while (0)
#define ONYX_ASSERT_EQ(a, b) do { if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b); } while (0)

ONYX_TEST(StringFilterSubstring) {
    std::string heap = "applebananaapricotavocadobig_apple";
    std::vector<onyx::kernels::StringOffset> offsets = {
        {0, 5},   // apple
        {5, 6},   // banana
        {11, 7},  // apricot
        {18, 7},  // avocado
        {25, 9}   // big_apple
    };

    std::vector<uint32_t> matched_ids(offsets.size());
    uint32_t match_count = 0;
    std::string pattern = "apple";

    onyx::kernels::launch_string_filter_contains(
        heap.data(), offsets.data(), offsets.size(),
        pattern.data(), pattern.size(),
        matched_ids.data(), &match_count, nullptr);

    // apple (idx 0) and big_apple (idx 4)
    ONYX_ASSERT_EQ(match_count, 2);
    ONYX_ASSERT_EQ(matched_ids[0], 0);
    ONYX_ASSERT_EQ(matched_ids[1], 4);
}
