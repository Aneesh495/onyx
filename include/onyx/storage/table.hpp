#pragma once

#include "onyx/storage/column_batch.hpp"
#include <string>
#include <vector>
#include <memory>
#include <mutex>

namespace onyx::storage {

class Table {
public:
    Table(std::string name, std::shared_ptr<Schema> schema);
    ~Table() = default;

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] std::shared_ptr<Schema> schema() const noexcept { return schema_; }
    [[nodiscard]] size_t num_rows() const noexcept;
    [[nodiscard]] size_t num_batches() const noexcept;

    void append_batch(std::shared_ptr<ColumnBatch> batch);
    [[nodiscard]] std::shared_ptr<ColumnBatch> batch(size_t idx) const;
    [[nodiscard]] std::vector<std::shared_ptr<ColumnBatch>> batches() const;

    // Create a table from CSV data
    static std::shared_ptr<Table> from_csv(std::string name, std::string_view csv_content, char delimiter = ',');

    // Render table as formatted text string for CLI
    [[nodiscard]] std::string to_formatted_string(size_t max_rows = 50) const;

private:
    std::string name_;
    std::shared_ptr<Schema> schema_;
    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<ColumnBatch>> batches_;
    size_t total_rows_{0};
};

} // namespace onyx::storage
