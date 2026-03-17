#include "onyx/server/json_serializer.hpp"
#include <sstream>

namespace onyx::server {

std::string JsonSerializer::serialize_batch(const storage::ColumnBatch& batch, size_t max_rows) {
    std::ostringstream ss;
    ss << "{\"schema\":[";

    auto schema = batch.schema();
    for (size_t c = 0; c < batch.num_columns(); ++c) {
        if (c > 0) ss << ",";
        ss << "{\"name\":\"" << schema->field(c).name << "\",\"type\":\"" << schema->field(c).type.to_string() << "\"}";
    }
    ss << "],\"rows\":[";

    size_t rows_to_emit = std::min(batch.num_rows(), max_rows);
    for (size_t r = 0; r < rows_to_emit; ++r) {
        if (r > 0) ss << ",";
        ss << "[";
        for (size_t c = 0; c < batch.num_columns(); ++c) {
            if (c > 0) ss << ",";
            auto val = batch.column(c)->get_scalar(r);
            if (val.is_null()) {
                ss << "null";
            } else if (val.type().id == TypeId::STRING) {
                ss << "\"" << val.to_string() << "\"";
            } else {
                ss << val.to_string();
            }
        }
        ss << "]";
    }
    ss << "],\"total_rows\":" << batch.num_rows() << "}";
    return ss.str();
}

std::string JsonSerializer::serialize_metrics(
    double latency_ms,
    double kernel_time_us,
    size_t total_rows,
    size_t output_rows)
{
    std::ostringstream ss;
    ss << "{\"latency_ms\":" << latency_ms
       << ",\"kernel_time_us\":" << kernel_time_us
       << ",\"input_rows\":" << total_rows
       << ",\"output_rows\":" << output_rows << "}";
    return ss.str();
}

} // namespace onyx::server
