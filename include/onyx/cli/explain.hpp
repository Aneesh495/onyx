#pragma once

#include <string>

namespace onyx::cli {

class QueryExplainer {
public:
    // Generate ASCII plan explanation for a given SQL query string
    static std::string explain_query(const std::string& sql, bool verbose = false);
};

} // namespace onyx::cli
