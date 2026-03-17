#include "test_framework.hpp"
#include "onyx/server/json_serializer.hpp"

ONYX_TEST(JsonSerializerRowBatch) {
    auto schema = std::make_shared<onyx::storage::Schema>(std::vector<onyx::storage::Field>{
        {"id", onyx::DataType(onyx::TypeId::INT32)},
        {"val", onyx::DataType(onyx::TypeId::FLOAT64)}
    });

    auto batch = std::make_shared<onyx::storage::ColumnBatch>(schema, 2);
    batch->column(0)->append_scalar(onyx::ScalarValue::make_int32(1));
    batch->column(0)->append_scalar(onyx::ScalarValue::make_int32(2));
    batch->column(1)->append_scalar(onyx::ScalarValue::make_float64(3.14));
    batch->column(1)->append_scalar(onyx::ScalarValue::make_float64(2.71));

    std::string json = onyx::server::JsonSerializer::serialize_batch(*batch);
    ONYX_ASSERT(!json.empty());
    ONYX_ASSERT(json.find("\"schema\":[") != std::string::npos);
    ONYX_ASSERT(json.find("\"rows\":[") != std::string::npos);
    ONYX_ASSERT(json.find("\"total_rows\":2") != std::string::npos);
}
