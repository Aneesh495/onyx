#include "test_framework.hpp"
#include "onyx/kernels/kernel_launchers.hpp"
#include <vector>

ONYX_TEST(BitpackUnpack4BitWords) {
    // 4-bit numbers packed into 64-bit words:
    // Word 0: 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x0
    uint64_t word = 0;
    for (uint64_t i = 0; i < 16; ++i) {
        word |= (i << (i * 4));
    }

    std::vector<uint64_t> packed_words = {word};
    std::vector<uint32_t> unpacked(16, 0);

    onyx::kernels::launch_bitpack_unpack(
        packed_words.data(), 4, 16, unpacked.data(), nullptr);

    for (uint32_t i = 0; i < 16; ++i) {
        ONYX_ASSERT_EQ(unpacked[i], i);
    }
}

ONYX_TEST(RleDecodeLongRuns) {
    // Run values: [42, 99, 7]
    // Run lengths: [100, 200, 50] -> cumulative [100, 300, 350]
    std::vector<int32_t> run_vals = {42, 99, 7};
    std::vector<uint32_t> cumulative_lens = {100, 300, 350};
    std::vector<int32_t> out_uncompressed(350, 0);

    onyx::kernels::launch_rle_decode<int32_t>(
        run_vals.data(), cumulative_lens.data(), run_vals.size(), 350, out_uncompressed.data(), nullptr);

    for (size_t i = 0; i < 100; ++i) ONYX_ASSERT_EQ(out_uncompressed[i], 42);
    for (size_t i = 100; i < 300; ++i) ONYX_ASSERT_EQ(out_uncompressed[i], 99);
    for (size_t i = 300; i < 350; ++i) ONYX_ASSERT_EQ(out_uncompressed[i], 7);
}
