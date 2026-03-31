#include "test_framework.hpp"
#include "onyx/storage/catalog.hpp"
#include "onyx/storage/table.hpp"

ONYX_TEST(CatalogRegisterListAndDrop) {
    auto& catalog = onyx::storage::Catalog::instance();
    catalog.clear();

    auto schema1 = std::make_shared<onyx::storage::Schema>(std::vector<onyx::storage::Field>{
        {"id", onyx::DataType(onyx::TypeId::INT32)}
    });
    auto t1 = std::make_shared<onyx::storage::Table>("users", schema1);
    auto t2 = std::make_shared<onyx::storage::Table>("transactions", schema1);

    ONYX_ASSERT(catalog.register_table(t1));
    ONYX_ASSERT(catalog.register_table(t2));

    ONYX_ASSERT(catalog.has_table("users"));
    ONYX_ASSERT(catalog.has_table("transactions"));
    ONYX_ASSERT(!catalog.has_table("nonexistent"));

    auto tables = catalog.list_tables();
    ONYX_ASSERT_EQ(tables.size(), 2);

    ONYX_ASSERT(catalog.drop_table("users"));
    ONYX_ASSERT(!catalog.has_table("users"));
    ONYX_ASSERT(catalog.has_table("transactions"));

    catalog.clear();
    ONYX_ASSERT_EQ(catalog.list_tables().size(), 0);
}
