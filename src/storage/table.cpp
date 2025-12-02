#include "onyx/storage/table.hpp"
#include <sstream>
#include <iomanip>

namespace onyx::storage {

Table::Table(std::string name, std::shared_ptr<Schema> schema)
    : name_(std::move(name)), schema_(std::move(schema)) {}

size_t Table::num_rows() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_rows_;
}

size_t Table::num_batches() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return batches_.size();
}

void Table::append_batch(std::shared_ptr<ColumnBatch> batch) {
    if (!batch) return;
    std::lock_guard<std::mutex> lock(mutex_);
    total_rows_ += batch->num_rows();
    batches_.push_back(std::move(batch));
}

std::shared_ptr<ColumnBatch> Table::batch(size_t idx) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (idx < batches_.size()) return batches_[idx];
    return nullptr;
}

std::vector<std::shared_ptr<ColumnBatch>> Table::batches() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return batches_;
}

std::shared_ptr<Table> Table::from_csv(std::string name, std::string_view csv_content, char delimiter) {
    std::istringstream stream{std::string(csv_content)};
    std::string line;

    // Header line
    if (!std::getline(stream, line)) {
        return nullptr;
    }

    std::vector<Field> fields;
    std::istringstream header_stream(line);
    std::string col_name;
    while (std::getline(header_stream, col_name, delimiter)) {
        // Trim whitespace
        size_t first = col_name.find_first_not_of(" \t\r\n");
        size_t last = col_name.find_last_not_of(" \t\r\n");
        if (first != std::string::npos && last != std::string::npos) {
            col_name = col_name.substr(first, last - first + 1);
        }
        fields.push_back(Field{.name = col_name, .type = DataType(TypeId::FLOAT64), .nullable = true});
    }

    auto schema = std::make_shared<Schema>(std::move(fields));
    auto table = std::make_shared<Table>(std::move(name), schema);
    auto current_batch = std::make_shared<ColumnBatch>(schema);

    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        std::istringstream row_stream(line);
        std::string token;
        size_t col_idx = 0;

        while (std::getline(row_stream, token, delimiter) && col_idx < schema->num_fields()) {
            try {
                double val = std::stod(token);
                current_batch->column(col_idx)->append_scalar(ScalarValue::make_float64(val));
            } catch (...) {
                current_batch->column(col_idx)->append_null();
            }
            col_idx++;
        }
    }

    if (schema->num_fields() > 0 && current_batch->column(0)) {
        current_batch->set_num_rows(current_batch->column(0)->size());
    }

    table->append_batch(std::move(current_batch));
    return table;
}

std::string Table::to_formatted_string(size_t max_rows) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;

    oss << "+";
    for (size_t i = 0; i < schema_->num_fields(); ++i) {
        oss << "------------------+";
    }
    oss << "\n|";
    for (size_t i = 0; i < schema_->num_fields(); ++i) {
        oss << " " << std::left << std::setw(16) << schema_->field(i).name << " |";
    }
    oss << "\n+";
    for (size_t i = 0; i < schema_->num_fields(); ++i) {
        oss << "------------------+";
    }
    oss << "\n";

    size_t rows_printed = 0;
    for (const auto& b : batches_) {
        for (size_t r = 0; r < b->num_rows(); ++r) {
            if (rows_printed >= max_rows) {
                oss << "| ... (more rows truncated) ...\n";
                break;
            }
            oss << "|";
            for (size_t c = 0; c < b->num_columns(); ++c) {
                std::string val = b->column(c)->get_scalar(r).to_string();
                if (val.size() > 16) val = val.substr(0, 13) + "...";
                oss << " " << std::right << std::setw(16) << val << " |";
            }
            oss << "\n";
            rows_printed++;
        }
        if (rows_printed >= max_rows) break;
    }

    oss << "+";
    for (size_t i = 0; i < schema_->num_fields(); ++i) {
        oss << "------------------+";
    }
    oss << "\n(" << total_rows_ << " rows total)\n";
    return oss.str();
}

} // namespace onyx::storage
