#include "onyx/kernels/kernel_launchers.hpp"
#include <vector>
#include <iostream>
#include <functional>
#include <string>

extern void register_benchmark(std::string name, size_t num_elements, std::function<void()> func);

static void bench_multi_column_sort() {
    constexpr size_t N = 500000;
    std::vector<int32_t> dept(N);
    std::vector<double> salary(N);
    for (size_t i = 0; i < N; ++i) {
        dept[i] = static_cast<int32_t>((i * 17) % 50);
        salary[i] = static_cast<double>(1000 + (i * 31) % 9000);
    }

    std::vector<onyx::kernels::SortKeyDescriptor> keys = {
        {
            .column_data = dept.data(),
            .type = onyx::DataType(onyx::TypeId::INT32),
            .direction = onyx::kernels::SortKeyDescriptor::Direction::Ascending,
            .nulls_first = false
        },
        {
            .column_data = salary.data(),
            .type = onyx::DataType(onyx::TypeId::FLOAT64),
            .direction = onyx::kernels::SortKeyDescriptor::Direction::Descending,
            .nulls_first = false
        }
    };

    std::vector<uint32_t> perm(N);
    onyx::kernels::launch_multi_column_sort(keys, N, perm.data(), nullptr);
}

int reg_multi_sort = (register_benchmark("RadixSort_MultiColumn_Composite", 500000, bench_multi_column_sort), 0);
