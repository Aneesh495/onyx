#pragma once

#include "onyx/storage/column.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace onyx::storage {

struct Field {
    std::string name;
    DataType type;
    bool nullable{true};

    bool operator==(const Field& other) const = default;
};

class Schema {
public:
    Schema() = default;
    explicit Schema(std::vector<Field> fields);

    [[nodiscard]] const std::vector<Field>& fields() const noexcept { return fields_; }
    [[nodiscard]] size_t num_fields() const noexcept { return fields_.size(); }
    [[nodiscard]] const Field& field(size_t idx) const { return fields_.at(idx); }

    [[nodiscard]] int find_field_idx(std::string_view name) const;
    [[nodiscard]] bool has_field(std::string_view name) const;

    [[nodiscard]] std::string to_string() const;

    bool operator==(const Schema& other) const = default;

private:
    std::vector<Field> fields_;
    std::unordered_map<std::string, size_t> name_to_idx_;
};

class ColumnBatch {
public:
    ColumnBatch(std::shared_ptr<Schema> schema, size_t num_rows = 0);
    ~ColumnBatch() = default;

    [[nodiscard]] std::shared_ptr<Schema> schema() const noexcept { return schema_; }
    [[nodiscard]] size_t num_columns() const noexcept { return columns_.size(); }
    [[nodiscard]] size_t num_rows() const noexcept { return num_rows_; }

    [[nodiscard]] std::shared_ptr<Column> column(size_t idx) const { return columns_.at(idx); }
    [[nodiscard]] std::shared_ptr<Column> column(std::string_view name) const;

    void set_column(size_t idx, std::shared_ptr<Column> col);
    void set_num_rows(size_t n) noexcept { num_rows_ = n; }

    // Memory residency
    void to_device(cudaStream_t stream = nullptr);
    void to_host(cudaStream_t stream = nullptr);
    [[nodiscard]] bool is_on_device() const;

    // Filter/slice batch by row IDs
    [[nodiscard]] std::shared_ptr<ColumnBatch> filter(const uint32_t* row_ids, size_t count, cudaStream_t stream = nullptr) const;

private:
    std::shared_ptr<Schema> schema_;
    std::vector<std::shared_ptr<Column>> columns_;
    size_t num_rows_{0};
};

} // namespace onyx::storage
