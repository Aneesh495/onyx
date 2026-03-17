#pragma once

#include "onyx/storage/column_batch.hpp"
#include <string>
#include <memory>

namespace onyx::server {

class JsonSerializer {
public:
    // Serialize ColumnBatch into JSON array of row objects
    static std::string serialize_batch(const storage::ColumnBatch& batch, size_t max_rows = 1000);

    // Serialize key-value metrics into JSON object
    static std::string serialize_metrics(
        double latency_ms,
        double kernel_time_us,
        size_t total_rows,
        size_t output_rows);
};

} // namespace onyx::server
