#include "onyx/execution/query_engine.hpp"
#include "onyx/storage/table.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <sstream>
#include <string>
#include <functional>

extern void register_benchmark(std::string name, size_t num_elements, std::function<void()> func);

static void bench_tpch_q6() {
    auto& engine = onyx::execution::QueryEngine::instance();

    // Create 100,000 synthetic lineitem rows
    std::ostringstream ss;
    ss << "shipdate,discount,quantity,extendedprice\n";
    for (int i = 0; i < 100000; ++i) {
        int date = 19940101 + (i % 365);
        double disc = 0.05 + (i % 5) * 0.01;
        double qty = 10.0 + (i % 30);
        double price = 500.0 + (i % 2000);
        ss << date << "," << disc << "," << qty << "," << price << "\n";
    }

    auto tbl = onyx::storage::Table::from_csv("lineitem_bench", ss.str());
    engine.catalog().register_table(tbl);

    auto res = engine.execute_query(
        "SELECT extendedprice FROM lineitem_bench WHERE quantity < 24;"
    );
}

int register_tpch_bench = (register_benchmark("TPCH_Q6_RevenueForecasting", 100000, bench_tpch_q6), 0);
