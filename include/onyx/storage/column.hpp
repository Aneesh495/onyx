#pragma once

#include "onyx/common/types.hpp"
#include "onyx/cuda/gpu_memory_pool.hpp"
#include <memory>
#include <vector>
#include <string>

namespace onyx::storage {

enum class CompressionCodec : uint8_t {
    None = 0,
    BitPacked,
    RunLength,
    Dictionary,
    FrameOfReference
};

class Column {
public:
    Column(DataType type, size_t capacity = 0, bool on_device = false);
    ~Column();

    Column(Column&& other) noexcept;
    Column& operator=(Column&& other) noexcept;

    Column(const Column&) = delete;
    Column& operator=(const Column&) = delete;

    [[nodiscard]] DataType type() const noexcept { return type_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool is_on_device() const noexcept { return on_device_; }
    [[nodiscard]] size_t null_count() const noexcept { return null_count_; }

    [[nodiscard]] void* raw_data() noexcept { return data_; }
    [[nodiscard]] const void* raw_data() const noexcept { return data_; }
    [[nodiscard]] uint8_t* null_bitmap() noexcept { return null_bitmap_; }
    [[nodiscard]] const uint8_t* null_bitmap() const noexcept { return null_bitmap_; }

    template <typename T>
    [[nodiscard]] T* data() noexcept { return static_cast<T*>(data_); }

    template <typename T>
    [[nodiscard]] const T* data() const noexcept { return static_cast<const T*>(data_); }

    void resize(size_t new_size);
    void reserve(size_t new_capacity);

    // Append scalar value
    void append_scalar(const ScalarValue& val);
    void append_null();

    // Device / Host memory transfer
    void to_device(cudaStream_t stream = nullptr);
    void to_host(cudaStream_t stream = nullptr);

    // Create a new column with gathered rows by indices
    [[nodiscard]] std::shared_ptr<Column> gather(const uint32_t* row_ids, size_t count, cudaStream_t stream = nullptr) const;

    // String storage for variable-length strings
    void add_string(std::string_view str);
    [[nodiscard]] std::string_view get_string(size_t row_idx) const;

    [[nodiscard]] ScalarValue get_scalar(size_t row_idx) const;

private:
    DataType type_;
    size_t size_{0};
    size_t capacity_{0};
    size_t null_count_{0};
    bool on_device_{false};

    void* data_{nullptr};
    uint8_t* null_bitmap_{nullptr};

    // Auxiliary buffers for variable-length strings
    std::vector<char> string_heap_;
    std::vector<uint32_t> string_offsets_;
    std::vector<uint32_t> string_lengths_;
};

} // namespace onyx::storage
