#pragma once

#include "onyx/execution/query_engine.hpp"
#include <string>

namespace onyx::cli {

class Repl {
public:
    explicit Repl(execution::QueryEngine* engine = &execution::QueryEngine::instance());
    ~Repl() = default;

    void run();
    void execute_single_query(std::string_view sql);
    void load_csv_file(const std::string& path, const std::string& table_name);

private:
    void print_banner();
    void print_help();
    void print_memory_stats();
    void print_tables();

    execution::QueryEngine* engine_;
};

} // namespace onyx::cli
