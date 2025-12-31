#include "onyx/cuda/cuda_graph.hpp"
#include <stdexcept>

extern void register_test(std::string name, std::function<void()> func);

#define ONYX_TEST(name) \
    static void test_##name(); \
    static int dummy_##name = (register_test(#name, test_##name), 0); \
    static void test_##name()

#define ONYX_ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while (0)
#define ONYX_ASSERT_EQ(a, b) do { if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b); } while (0)

ONYX_TEST(CudaGraphCaptureAndReplay) {
    onyx::cuda::CudaGraphExecutor executor;
    executor.begin_capture(nullptr);
    // Sequence of stream operations
    executor.end_capture(nullptr);

    ONYX_ASSERT(executor.is_instantiated());
    // Replay execution without re-launching host driver nodes
    executor.launch(nullptr);
}
