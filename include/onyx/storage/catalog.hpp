#pragma once

#include "onyx/storage/table.hpp"
#include <unordered_map>
#include <shared_mutex>
#include <string>
#include <vector>
#include <memory>

namespace onyx::storage {

class Catalog {
public:
    static Catalog& instance() {
        static Catalog cat;
        return cat;
    }

    Catalog() = default;
    ~Catalog() = default;

    bool register_table(std::shared_ptr<Table> table);
    [[nodiscard]] std::shared_ptr<Table> get_table(std::string_view name) const;
    bool drop_table(std::string_view name);
    [[nodiscard]] bool has_table(std::string_view name) const;
    [[nodiscard]] std::vector<std::string> list_tables() const;

    void clear();

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<Table>> tables_;
};

} // namespace onyx::storage
