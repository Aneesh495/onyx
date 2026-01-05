#include "test_framework.hpp"
#include "onyx/cuda/cuda_graph.hpp"
#include <stdexcept>


ONYX_TEST(CudaGraphCaptureAndReplay) {
    onyx::cuda::CudaGraphExecutor executor;
    executor.begin_capture(nullptr);
    // Sequence of stream operations
    executor.end_capture(nullptr);

    ONYX_ASSERT(executor.is_instantiated());
    // Replay execution without re-launching host driver nodes
    executor.launch(nullptr);
}
