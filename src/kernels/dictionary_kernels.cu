#include "onyx/kernels/dictionary.cuh"

namespace onyx::kernels {

template <typename T>
__global__ void dictionary_decode_cuda_kernel(
    const uint32_t* __restrict__ encoded_codes,
    const T* __restrict__ dictionary_entries,
    size_t num_dict_entries,
    size_t num_rows,
    T* __restrict__ out_decoded)
{
    dictionary_decode_kernel<T>(
        encoded_codes, dictionary_entries, num_dict_entries, num_rows, out_decoded);
}

} // namespace onyx::kernels
