#include "onyx/kernels/compression.cuh"
#include <vector>
#include <stdexcept>

namespace onyx::kernels {
void launch_bitpack_unpack(const uint64_t* packed_words, uint32_t bit_width, size_t num_elements, uint32_t* out_unpacked, cudaStream_t stream);
void launch_for_decode(int64_t base_value, const int32_t* deltas, size_t num_elements, int64_t* out_values, cudaStream_t stream);
template <typename T>
void launch_rle_decode(const T* run_values, const uint32_t* run_cumulative_lengths, size_t num_runs, size_t total_elements, T* out_uncompressed, cudaStream_t stream);
}

extern void register_test(std::string name, std::function<void()> func);

#define ONYX_TEST(name) \
    static void test_##name(); \
    static int dummy_##name = (register_test(#name, test_##name), 0); \
    static void test_##name()

#define ONYX_ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while (0)
#define ONYX_ASSERT_EQ(a, b) do { if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b); } while (0)

ONYX_TEST(FrameOfReferenceDecode) {
    int64_t base = 1000;
    std::vector<int32_t> deltas = {5, 10, -3, 22};
    std::vector<int64_t> out(deltas.size());

    onyx::kernels::launch_for_decode(base, deltas.data(), deltas.size(), out.data(), nullptr);

    ONYX_ASSERT_EQ(out[0], 1005);
    ONYX_ASSERT_EQ(out[1], 1010);
    ONYX_ASSERT_EQ(out[2], 997);
    ONYX_ASSERT_EQ(out[3], 1022);
}

ONYX_TEST(RunLengthEncodingDecode) {
    std::vector<int32_t> run_vals = {7, 42, 99};
    std::vector<uint32_t> run_offsets = {3, 7, 10}; // 3 x 7, 4 x 42, 3 x 99
    std::vector<int32_t> out(10);

    onyx::kernels::launch_rle_decode<int32_t>(
        run_vals.data(), run_offsets.data(), run_vals.size(), 10, out.data(), nullptr);

    ONYX_ASSERT_EQ(out[0], 7);
    ONYX_ASSERT_EQ(out[2], 7);
    ONYX_ASSERT_EQ(out[3], 42);
    ONYX_ASSERT_EQ(out[6], 42);
    ONYX_ASSERT_EQ(out[7], 99);
    ONYX_ASSERT_EQ(out[9], 99);
}
