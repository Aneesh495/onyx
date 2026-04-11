#include "onyx/cli/repl.hpp"
#include "onyx/cli/explain.hpp"
#include "onyx/storage/table.hpp"
#include <iostream>

int main(int argc, char** argv) {
    // Populate default synthetic demo dataset: telemetry
    std::string demo_csv =
        "device_id,latency_us,temperature,status\n"
        "101,120.5,45.2,1\n"
        "102,1540.2,68.1,2\n"
        "103,450.8,52.3,1\n"
        "104,2100.0,74.5,3\n"
        "105,89.1,41.0,1\n"
        "106,1620.4,70.2,2\n"
        "107,310.0,49.8,1\n"
        "108,1890.7,72.4,3\n";

    auto table = onyx::storage::Table::from_csv("telemetry", demo_csv);
    onyx::storage::Catalog::instance().register_table(table);

    onyx::cli::Repl repl;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--exec" || arg == "-e") && i + 1 < argc) {
            repl.execute_single_query(argv[++i]);
            return 0;
        } else if (arg == "--explain" && i + 1 < argc) {
            std::cout << onyx::cli::QueryExplainer::explain_query(argv[++i], true) << "\n";
            return 0;
        } else if (arg == "--load" && i + 2 < argc) {
            std::string path = argv[++i];
            std::string tbl = argv[++i];
            repl.load_csv_file(path, tbl);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: onyx-cli [OPTIONS]\n"
                      << "  -e, --exec <query>   Execute single query and exit\n"
                      << "  --explain <query>    Explain query execution plan\n"
                      << "  --load <file> <tbl>  Load CSV file before starting\n"
                      << "  --help               Display this help message\n";
            return 0;
        }
    }

    repl.run();
    return 0;
}
