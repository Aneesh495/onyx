#include "onyx/storage/arrow_adapter.hpp"
#include <cstdlib>
#include <cstring>

namespace onyx::storage {

static void release_arrow_schema(ArrowSchema* schema) {
    if (!schema || !schema->release) return;
    if (schema->name && schema->name[0] != '\0') {
        free(const_cast<char*>(schema->name));
        schema->name = nullptr;
    }
    if (schema->children) {
        for (int64_t i = 0; i < schema->n_children; ++i) {
            if (schema->children[i]) {
                if (schema->children[i]->release) schema->children[i]->release(schema->children[i]);
                delete schema->children[i];
            }
        }
        delete[] schema->children;
    }
    schema->release = nullptr;
}

static void release_arrow_array(ArrowArray* array) {
    if (!array || !array->release) return;
    if (array->buffers) {
        delete[] array->buffers;
    }
    if (array->children) {
        for (int64_t i = 0; i < array->n_children; ++i) {
            if (array->children[i]) {
                if (array->children[i]->release) array->children[i]->release(array->children[i]);
                delete array->children[i];
            }
        }
        delete[] array->children;
    }
    array->release = nullptr;
}

static const char* type_to_arrow_format(TypeId type) {
    switch (type) {
        case TypeId::INT8: return "c";
        case TypeId::INT16: return "s";
        case TypeId::INT32: return "i";
        case TypeId::INT64: return "l";
        case TypeId::FLOAT32: return "f";
        case TypeId::FLOAT64: return "g";
        case TypeId::STRING: return "u";
        case TypeId::TIMESTAMP: return "tsn:";
        default: return "i";
    }
}

void ArrowAdapter::export_batch(
    const ColumnBatch& batch,
    ArrowSchema* out_schema,
    ArrowArray* out_array)
{
    if (!out_schema || !out_array) return;

    size_t num_cols = batch.num_columns();
    size_t num_rows = batch.num_rows();

    // Export root schema
    out_schema->format = "+s";
    out_schema->name = "";
    out_schema->metadata = nullptr;
    out_schema->flags = 0;
    out_schema->n_children = static_cast<int64_t>(num_cols);
    out_schema->children = new ArrowSchema*[num_cols];
    out_schema->dictionary = nullptr;
    out_schema->release = release_arrow_schema;

    // Export root array
    out_array->length = static_cast<int64_t>(num_rows);
    out_array->null_count = 0;
    out_array->offset = 0;
    out_array->n_buffers = 1;
    out_array->n_children = static_cast<int64_t>(num_cols);
    out_array->buffers = new const void*[1]{nullptr};
    out_array->children = new ArrowArray*[num_cols];
    out_array->dictionary = nullptr;
    out_array->release = release_arrow_array;

    for (size_t c = 0; c < num_cols; ++c) {
        auto col = batch.column(c);
        auto field = batch.schema()->field(c);

        // Child schema
        auto child_schema = new ArrowSchema();
        child_schema->format = type_to_arrow_format(field.type.id);
        child_schema->name = strdup(field.name.c_str());
        child_schema->metadata = nullptr;
        child_schema->flags = field.nullable ? 2 : 0;
        child_schema->n_children = 0;
        child_schema->children = nullptr;
        child_schema->dictionary = nullptr;
        child_schema->release = release_arrow_schema;
        out_schema->children[c] = child_schema;

        // Child array
        auto child_array = new ArrowArray();
        child_array->length = static_cast<int64_t>(num_rows);
        child_array->null_count = static_cast<int64_t>(col->null_count());
        child_array->offset = 0;
        child_array->n_buffers = 2;
        child_array->n_children = 0;
        child_array->children = nullptr;
        child_array->dictionary = nullptr;
        child_array->release = release_arrow_array;

        // Zero-copy pointer exposure: bitmap + raw data
        const void** bufs = new const void*[2];
        bufs[0] = col->null_bitmap();
        bufs[1] = col->raw_data();
        child_array->buffers = bufs;

        out_array->children[c] = child_array;
    }
}

std::shared_ptr<ColumnBatch> ArrowAdapter::import_batch(
    ArrowSchema* schema,
    ArrowArray* array)
{
    if (!schema || !array) return nullptr;

    std::vector<Field> fields;
    size_t num_cols = static_cast<size_t>(schema->n_children);
    size_t num_rows = static_cast<size_t>(array->length);

    for (size_t c = 0; c < num_cols; ++c) {
        auto child = schema->children[c];
        std::string fname = child->name ? child->name : ("col_" + std::to_string(c));
        fields.push_back(Field{
            .name = fname,
            .type = DataType(TypeId::FLOAT64),
            .nullable = (child->flags & 2) != 0
        });
    }

    auto out_schema = std::make_shared<Schema>(std::move(fields));
    auto batch = std::make_shared<ColumnBatch>(out_schema, num_rows);
    return batch;
}

} // namespace onyx::storage
