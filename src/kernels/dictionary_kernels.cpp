#include "onyx/kernels/dictionary.cuh"
#include "onyx/cuda/simt_engine.hpp"
#include <algorithm>

namespace onyx::kernels {

template <typename T>
void launch_dictionary_decode(
    const uint32_t* encoded_codes,
    const T* dictionary_entries,
    size_t num_dict_entries,
    size_t num_rows,
    T* out_decoded,
    cudaStream_t stream)
{
    if (num_rows == 0 || num_dict_entries == 0) return;

#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
    uint32_t block_size = 256;
    uint32_t grid_size = std::min<uint32_t>((num_rows + block_size - 1) / block_size, 1024);
    // CUDA kernel launch
#else
    for (size_t i = 0; i < num_rows; ++i) {
        uint32_t code = encoded_codes[i];
        if (code < num_dict_entries) {
            out_decoded[i] = dictionary_entries[code];
        }
    }
#endif
}

// Instantiations
template void launch_dictionary_decode<int32_t>(const uint32_t*, const int32_t*, size_t, size_t, int32_t*, cudaStream_t);
template void launch_dictionary_decode<int64_t>(const uint32_t*, const int64_t*, size_t, size_t, int64_t*, cudaStream_t);
template void launch_dictionary_decode<float>(const uint32_t*, const float*, size_t, size_t, float*, cudaStream_t);
template void launch_dictionary_decode<double>(const uint32_t*, const double*, size_t, size_t, double*, cudaStream_t);

} // namespace onyx::kernels
