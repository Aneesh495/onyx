#include "test_framework.hpp"
#include "onyx/storage/arrow_adapter.hpp"
#include <cstring>

ONYX_TEST(ArrowCDataInterfaceZeroCopyExport) {
    auto schema = std::make_shared<onyx::storage::Schema>(std::vector<onyx::storage::Field>{
        {"sensor_id", onyx::DataType(onyx::TypeId::INT32)},
        {"temperature", onyx::DataType(onyx::TypeId::FLOAT64)}
    });

    auto batch = std::make_shared<onyx::storage::ColumnBatch>(schema, 3);
    batch->column(0)->append_scalar(onyx::ScalarValue::make_int32(101));
    batch->column(0)->append_scalar(onyx::ScalarValue::make_int32(102));
    batch->column(0)->append_scalar(onyx::ScalarValue::make_int32(103));

    batch->column(1)->append_scalar(onyx::ScalarValue::make_float64(22.5));
    batch->column(1)->append_scalar(onyx::ScalarValue::make_float64(23.0));
    batch->column(1)->append_scalar(onyx::ScalarValue::make_float64(24.5));

    onyx::storage::ArrowSchema arrow_schema;
    onyx::storage::ArrowArray arrow_array;

    onyx::storage::ArrowAdapter::export_batch(*batch, &arrow_schema, &arrow_array);

    ONYX_ASSERT_EQ(arrow_schema.n_children, 2);
    ONYX_ASSERT_EQ(arrow_array.length, 3);
    ONYX_ASSERT(std::strcmp(arrow_schema.format, "+s") == 0);

    // Verify first child column metadata and pointers
    ONYX_ASSERT(arrow_schema.children[0] != nullptr);
    ONYX_ASSERT(std::strcmp(arrow_schema.children[0]->name, "sensor_id") == 0);
    ONYX_ASSERT_EQ(arrow_array.children[0]->length, 3);

    // Release structures through Arrow C Data release callbacks
    if (arrow_schema.release) arrow_schema.release(&arrow_schema);
    if (arrow_array.release) arrow_array.release(&arrow_array);
}
