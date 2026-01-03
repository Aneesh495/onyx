#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <iomanip>

struct BenchmarkCase {
    std::string name;
    size_t num_elements;
    std::function<void()> func;
};

static std::vector<BenchmarkCase>& get_bench_registry() {
    static std::vector<BenchmarkCase> registry;
    return registry;
}

void register_benchmark(std::string name, size_t num_elements, std::function<void()> func) {
    get_bench_registry().push_back(BenchmarkCase{std::move(name), num_elements, std::move(func)});
}

int main() {
    auto& benchmarks = get_bench_registry();
    std::cout << "\033[1;36m========================================================================\033[0m\n";
    std::cout << "\033[1;36m           Onyx GPU & SIMT High-Throughput Benchmark Suite              \033[0m\n";
    std::cout << "\033[1;36m========================================================================\033[0m\n\n";

    std::cout << std::left << std::setw(32) << "Benchmark Kernel"
              << std::right << std::setw(12) << "Elements"
              << std::setw(14) << "Latency (ms)"
              << std::setw(18) << "Throughput (M/s)" << "\n";
    std::cout << "------------------------------------------------------------------------\n";

    for (const auto& bench : benchmarks) {
        // Warmup
        bench.func();

        constexpr int ITERATIONS = 3;
        double min_ms = 1e9;

        for (int it = 0; it < ITERATIONS; ++it) {
            auto start = std::chrono::high_resolution_clock::now();
            bench.func();
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (ms < min_ms) min_ms = ms;
        }

        double m_elements_per_sec = (static_cast<double>(bench.num_elements) / (min_ms * 1000.0));

        std::cout << std::left << std::setw(32) << bench.name
                  << std::right << std::setw(12) << bench.num_elements
                  << std::setw(14) << std::fixed << std::setprecision(2) << min_ms
                  << std::setw(18) << std::fixed << std::setprecision(2) << m_elements_per_sec
                  << "\n";
    }

    std::cout << "------------------------------------------------------------------------\n\n";
    return 0;
}
