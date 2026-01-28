#include "test_framework.hpp"
#include "onyx/kernels/kernel_launchers.hpp"
#include <vector>

ONYX_TEST(DictionaryDecodeBasic) {
    std::vector<int32_t> dictionary = {100, 200, 300, 400};
    std::vector<uint32_t> codes = {0, 2, 1, 3, 0, 1, 2, 2};
    std::vector<int32_t> decoded(codes.size(), 0);

    onyx::kernels::launch_dictionary_decode<int32_t>(
        codes.data(), dictionary.data(), dictionary.size(), codes.size(), decoded.data(), nullptr);

    std::vector<int32_t> expected = {100, 300, 200, 400, 100, 200, 300, 300};
    for (size_t i = 0; i < codes.size(); ++i) {
        ONYX_ASSERT_EQ(decoded[i], expected[i]);
    }
}
