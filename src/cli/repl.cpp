#include "onyx/cli/repl.hpp"
#include "onyx/cuda/gpu_memory_pool.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

namespace onyx::cli {

Repl::Repl(execution::QueryEngine* engine) : engine_(engine) {}

void Repl::print_banner() {
    std::cout << "\033[1;36m"
              << "   ____  _   ___  ___  __\n"
              << "  / __ \\/ | / / \\/ / |/ /\n"
              << " / /_/ /  |/ / \\  /|   / \n"
              << " \\____/_/|___/  /_/ /_/|_| \n"
              << "\033[0m"
              << " Onyx GPU Columnar Query & Streaming Kernel v0.1.0\n"
              << " Type \\help for commands, \\tables for catalog, \\q to exit.\n\n";
}

void Repl::print_help() {
    std::cout << "Available Commands:\n"
              << "  \\tables               List registered tables and schemas\n"
              << "  \\mem                  Display GPU memory pool occupancy and stats\n"
              << "  \\load <path> <table>  Load CSV dataset into table\n"
              << "  \\clear                Clear terminal screen\n"
              << "  \\help                 Show this help text\n"
              << "  \\q                    Exit Onyx CLI\n"
              << "  EXPLAIN <query>;      Print logical and optimized GPU physical plans\n\n";
}

void Repl::print_memory_stats() {
    auto stats = cuda::GpuMemoryPool::global().get_stats();
    std::cout << stats.to_string() << "\n\n";
}

void Repl::print_tables() {
    auto names = engine_->catalog().list_tables();
    if (names.empty()) {
        std::cout << "No tables loaded in catalog.\n\n";
        return;
    }
    std::cout << "Catalog Tables (" << names.size() << "):\n";
    for (const auto& name : names) {
        auto tbl = engine_->catalog().get_table(name);
        std::cout << "  * " << name << " (" << tbl->num_rows() << " rows, "
                  << tbl->num_batches() << " batches) "
                  << tbl->schema()->to_string() << "\n";
    }
    std::cout << "\n";
}

void Repl::load_csv_file(const std::string& path, const std::string& table_name) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << path << "\n";
        return;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    auto table = storage::Table::from_csv(table_name, ss.str());
    if (table) {
        engine_->catalog().register_table(table);
        std::cout << "Successfully loaded " << table->num_rows() << " rows into table '" << table_name << "'.\n\n";
    } else {
        std::cerr << "Error: Failed to parse CSV file.\n";
    }
}

void Repl::execute_single_query(std::string_view sql) {
    std::string trimmed(sql);
    size_t first = trimmed.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return;
    trimmed = trimmed.substr(first);

    if (trimmed.starts_with("EXPLAIN") || trimmed.starts_with("explain")) {
        std::cout << engine_->explain_query(trimmed.substr(7)) << "\n";
        return;
    }

    auto res = engine_->execute_query(sql);
    std::cout << res.to_formatted_output() << "\n";
}

void Repl::run() {
    print_banner();
    std::string line;
    std::string query_buffer;

    while (true) {
        if (query_buffer.empty()) {
            std::cout << "\033[1;32monyx>\033[0m ";
        } else {
            std::cout << "\033[1;32m   ->\033[0m ";
        }

        if (!std::getline(std::cin, line)) {
            break;
        }

        if (line == "\\q" || line == "quit" || line == "exit") {
            break;
        } else if (line == "\\help") {
            print_help();
            continue;
        } else if (line == "\\tables") {
            print_tables();
            continue;
        } else if (line == "\\mem") {
            print_memory_stats();
            continue;
        } else if (line == "\\clear") {
            std::cout << "\033[2J\033[1;1H";
            continue;
        } else if (line.starts_with("\\load ")) {
            std::istringstream iss(line.substr(6));
            std::string path, tbl;
            iss >> path >> tbl;
            if (!path.empty() && !tbl.empty()) {
                load_csv_file(path, tbl);
            } else {
                std::cout << "Usage: \\load <filepath> <tablename>\n";
            }
            continue;
        }

        query_buffer += " " + line;
        if (line.find(';') != std::string::npos) {
            execute_single_query(query_buffer);
            query_buffer.clear();
        }
    }
}

} // namespace onyx::cli
