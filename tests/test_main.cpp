#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <chrono>

struct TestCase {
    std::string name;
    std::function<void()> func;
};

static std::vector<TestCase>& get_registry() {
    static std::vector<TestCase> registry;
    return registry;
}

void register_test(std::string name, std::function<void()> func) {
    get_registry().push_back(TestCase{std::move(name), std::move(func)});
}

#define ONYX_TEST(name) \
    static void test_##name(); \
    static int dummy_##name = (register_test(#name, test_##name), 0); \
    static void test_##name()

#define ONYX_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            throw std::runtime_error("Assertion failed: " #cond); \
        } \
    } while (0)

#define ONYX_ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            std::cerr << "Assertion failed: " << #a << " == " << #b << " (" << (a) << " != " << (b) << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
            throw std::runtime_error("Assertion failed: " #a " == " #b); \
        } \
    } while (0)

int main() {
    auto& tests = get_registry();
    std::cout << "\033[1;36m========================================================\033[0m\n";
    std::cout << "\033[1;36m       Onyx Automated Test & Verification Suite         \033[0m\n";
    std::cout << "\033[1;36m========================================================\033[0m\n\n";

    size_t passed = 0;
    size_t failed = 0;

    auto total_start = std::chrono::high_resolution_clock::now();

    for (const auto& test : tests) {
        std::cout << "  [ RUN      ] " << test.name << std::flush;
        auto start = std::chrono::high_resolution_clock::now();
        try {
            test.func();
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << "\r  \033[1;32m[       OK ]\033[0m " << test.name << " (" << ms << " ms)\n";
            passed++;
        } catch (const std::exception& ex) {
            std::cout << "\r  \033[1;31m[  FAILED  ]\033[0m " << test.name << ": " << ex.what() << "\n";
            failed++;
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << "Total Tests Run: " << tests.size() << "\n";
    std::cout << "Passed: \033[1;32m" << passed << "\033[0m, Failed: "
              << (failed > 0 ? "\033[1;31m" : "\033[0m") << failed << "\033[0m\n";
    std::cout << "Elapsed Time:    " << total_ms << " ms\n";
    std::cout << "--------------------------------------------------------\n";

    return failed > 0 ? 1 : 0;
}
