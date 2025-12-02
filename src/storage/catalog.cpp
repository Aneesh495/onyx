#include "onyx/storage/catalog.hpp"

namespace onyx::storage {

bool Catalog::register_table(std::shared_ptr<Table> table) {
    if (!table) return false;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    tables_[table->name()] = std::move(table);
    return true;
}

std::shared_ptr<Table> Catalog::get_table(std::string_view name) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = tables_.find(std::string(name));
    if (it != tables_.end()) {
        return it->second;
    }
    return nullptr;
}

bool Catalog::drop_table(std::string_view name) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    return tables_.erase(std::string(name)) > 0;
}

bool Catalog::has_table(std::string_view name) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return tables_.contains(std::string(name));
}

std::vector<std::string> Catalog::list_tables() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(tables_.size());
    for (const auto& [name, _] : tables_) {
        names.push_back(name);
    }
    return names;
}

void Catalog::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    tables_.clear();
}

} // namespace onyx::storage
