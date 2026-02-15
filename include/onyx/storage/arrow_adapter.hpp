#pragma once

#include "onyx/storage/column_batch.hpp"
#include <cstdint>
#include <memory>
#include <string>

namespace onyx::storage {

// Apache Arrow C Data Interface layout specifications
struct ArrowSchema {
    const char* format{nullptr};
    const char* name{nullptr};
    const char* metadata{nullptr};
    int64_t flags{0};
    int64_t n_children{0};
    ArrowSchema** children{nullptr};
    ArrowSchema* dictionary{nullptr};
    void (*release)(ArrowSchema*){nullptr};
    void* private_data{nullptr};
};

struct ArrowArray {
    int64_t length{0};
    int64_t null_count{0};
    int64_t offset{0};
    int64_t n_buffers{0};
    int64_t n_children{0};
    const void** buffers{nullptr};
    ArrowArray** children{nullptr};
    ArrowArray* dictionary{nullptr};
    void (*release)(ArrowArray*){nullptr};
    void* private_data{nullptr};
};

class ArrowAdapter {
public:
    // Export ColumnBatch into Arrow C Data Interface structures
    static void export_batch(
        const ColumnBatch& batch,
        ArrowSchema* out_schema,
        ArrowArray* out_array);

    // Import Arrow C Data Interface structures into an Onyx ColumnBatch
    static std::shared_ptr<ColumnBatch> import_batch(
        ArrowSchema* schema,
        ArrowArray* array);
};

} // namespace onyx::storage
