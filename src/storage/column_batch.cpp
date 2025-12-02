#include "onyx/storage/column_batch.hpp"
#include <sstream>

namespace onyx::storage {

Schema::Schema(std::vector<Field> fields) : fields_(std::move(fields)) {
    for (size_t i = 0; i < fields_.size(); ++i) {
        name_to_idx_[fields_[i].name] = i;
    }
}

int Schema::find_field_idx(std::string_view name) const {
    auto it = name_to_idx_.find(std::string(name));
    if (it != name_to_idx_.end()) {
        return static_cast<int>(it->second);
    }
    return -1;
}

bool Schema::has_field(std::string_view name) const {
    return find_field_idx(name) != -1;
}

std::string Schema::to_string() const {
    std::ostringstream oss;
    oss << "Schema [";
    for (size_t i = 0; i < fields_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << fields_[i].name << ": " << fields_[i].type.to_string();
    }
    oss << "]";
    return oss.str();
}

ColumnBatch::ColumnBatch(std::shared_ptr<Schema> schema, size_t num_rows)
    : schema_(std::move(schema)), num_rows_(num_rows) {
    if (schema_) {
        columns_.reserve(schema_->num_fields());
        for (const auto& field : schema_->fields()) {
            columns_.push_back(std::make_shared<Column>(field.type, num_rows_));
        }
    }
}

std::shared_ptr<Column> ColumnBatch::column(std::string_view name) const {
    int idx = schema_->find_field_idx(name);
    if (idx < 0) return nullptr;
    return columns_[idx];
}

void ColumnBatch::set_column(size_t idx, std::shared_ptr<Column> col) {
    if (idx < columns_.size()) {
        columns_[idx] = std::move(col);
    }
}

void ColumnBatch::to_device(cudaStream_t stream) {
    for (auto& col : columns_) {
        if (col) col->to_device(stream);
    }
}

void ColumnBatch::to_host(cudaStream_t stream) {
    for (auto& col : columns_) {
        if (col) col->to_host(stream);
    }
}

bool ColumnBatch::is_on_device() const {
    if (columns_.empty() || !columns_[0]) return false;
    return columns_[0]->is_on_device();
}

std::shared_ptr<ColumnBatch> ColumnBatch::filter(const uint32_t* row_ids, size_t count, cudaStream_t stream) const {
    auto res = std::make_shared<ColumnBatch>(schema_, count);
    for (size_t i = 0; i < columns_.size(); ++i) {
        if (columns_[i]) {
            res->set_column(i, columns_[i]->gather(row_ids, count, stream));
        }
    }
    return res;
}

} // namespace onyx::storage
