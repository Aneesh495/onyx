#include "onyx/storage/column.hpp"
#include "onyx/common/bit_utils.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace onyx::storage {

Column::Column(DataType type, size_t capacity, bool on_device)
    : type_(type), capacity_(std::max(capacity, static_cast<size_t>(16))), on_device_(on_device) {
    size_t byte_size = capacity_ * type_.width;
    size_t bitmap_bytes = (capacity_ + 7) / 8;

    if (on_device_) {
        data_ = cuda::GpuMemoryPool::global().allocate(byte_size);
        null_bitmap_ = static_cast<uint8_t*>(cuda::GpuMemoryPool::global().allocate(bitmap_bytes));
        cudaMemset(null_bitmap_, 0xFF, bitmap_bytes);
    } else {
        posix_memalign(&data_, 128, std::max(byte_size, static_cast<size_t>(128)));
        posix_memalign(reinterpret_cast<void**>(&null_bitmap_), 128, std::max(bitmap_bytes, static_cast<size_t>(128)));
        std::memset(null_bitmap_, 0xFF, bitmap_bytes);
    }
}

Column::~Column() {
    if (data_) {
        if (on_device_) {
            cuda::GpuMemoryPool::global().deallocate(data_, capacity_ * type_.width);
            if (null_bitmap_) {
                cuda::GpuMemoryPool::global().deallocate(null_bitmap_, (capacity_ + 7) / 8);
            }
        } else {
            free(data_);
            if (null_bitmap_) free(null_bitmap_);
        }
        data_ = nullptr;
        null_bitmap_ = nullptr;
    }
}

Column::Column(Column&& other) noexcept
    : type_(other.type_), size_(other.size_), capacity_(other.capacity_),
      null_count_(other.null_count_), on_device_(other.on_device_),
      data_(other.data_), null_bitmap_(other.null_bitmap_),
      string_heap_(std::move(other.string_heap_)),
      string_offsets_(std::move(other.string_offsets_)),
      string_lengths_(std::move(other.string_lengths_)) {
    other.data_ = nullptr;
    other.null_bitmap_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
}

Column& Column::operator=(Column&& other) noexcept {
    if (this != &other) {
        if (data_) {
            if (on_device_) {
                cuda::GpuMemoryPool::global().deallocate(data_, capacity_ * type_.width);
                if (null_bitmap_) cuda::GpuMemoryPool::global().deallocate(null_bitmap_, (capacity_ + 7) / 8);
            } else {
                free(data_);
                if (null_bitmap_) free(null_bitmap_);
            }
        }
        type_ = other.type_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        null_count_ = other.null_count_;
        on_device_ = other.on_device_;
        data_ = other.data_;
        null_bitmap_ = other.null_bitmap_;
        string_heap_ = std::move(other.string_heap_);
        string_offsets_ = std::move(other.string_offsets_);
        string_lengths_ = std::move(other.string_lengths_);

        other.data_ = nullptr;
        other.null_bitmap_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    return *this;
}

void Column::reserve(size_t new_capacity) {
    if (new_capacity <= capacity_) return;

    size_t byte_size = new_capacity * type_.width;
    size_t bitmap_bytes = (new_capacity + 7) / 8;

    void* new_data = nullptr;
    uint8_t* new_bitmap = nullptr;

    if (on_device_) {
        new_data = cuda::GpuMemoryPool::global().allocate(byte_size);
        new_bitmap = static_cast<uint8_t*>(cuda::GpuMemoryPool::global().allocate(bitmap_bytes));
        if (size_ > 0) {
            cudaMemcpy(new_data, data_, size_ * type_.width, cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_bitmap, null_bitmap_, (size_ + 7) / 8, cudaMemcpyDeviceToDevice);
        }
        cuda::GpuMemoryPool::global().deallocate(data_, capacity_ * type_.width);
        cuda::GpuMemoryPool::global().deallocate(null_bitmap_, (capacity_ + 7) / 8);
    } else {
        posix_memalign(&new_data, 128, std::max(byte_size, static_cast<size_t>(128)));
        posix_memalign(reinterpret_cast<void**>(&new_bitmap), 128, std::max(bitmap_bytes, static_cast<size_t>(128)));
        std::memset(new_bitmap, 0xFF, bitmap_bytes);
        if (size_ > 0) {
            std::memcpy(new_data, data_, size_ * type_.width);
            std::memcpy(new_bitmap, null_bitmap_, (size_ + 7) / 8);
        }
        free(data_);
        free(null_bitmap_);
    }

    data_ = new_data;
    null_bitmap_ = new_bitmap;
    capacity_ = new_capacity;
}

void Column::resize(size_t new_size) {
    reserve(new_size);
    size_ = new_size;
}

void Column::append_scalar(const ScalarValue& val) {
    if (val.is_null()) {
        append_null();
        return;
    }

    if (size_ >= capacity_) {
        reserve(std::max(capacity_ * 2, static_cast<size_t>(64)));
    }

    if (type_.id == TypeId::STRING) {
        add_string(val.get<std::string>());
        return;
    }

    // Fixed width write
    char* dst = static_cast<char*>(data_) + (size_ * type_.width);
    switch (type_.id) {
        case TypeId::INT8:    *reinterpret_cast<int8_t*>(dst) = val.get<int8_t>(); break;
        case TypeId::INT16:   *reinterpret_cast<int16_t*>(dst) = val.get<int16_t>(); break;
        case TypeId::INT32:   *reinterpret_cast<int32_t*>(dst) = val.get<int32_t>(); break;
        case TypeId::INT64:   *reinterpret_cast<int64_t*>(dst) = val.get<int64_t>(); break;
        case TypeId::UINT8:   *reinterpret_cast<uint8_t*>(dst) = val.get<uint8_t>(); break;
        case TypeId::UINT16:  *reinterpret_cast<uint16_t*>(dst) = val.get<uint16_t>(); break;
        case TypeId::UINT32:  *reinterpret_cast<uint32_t*>(dst) = val.get<uint32_t>(); break;
        case TypeId::UINT64:  *reinterpret_cast<uint64_t*>(dst) = val.get<uint64_t>(); break;
        case TypeId::FLOAT32: *reinterpret_cast<float*>(dst) = val.get<float>(); break;
        case TypeId::FLOAT64: *reinterpret_cast<double*>(dst) = val.get<double>(); break;
        case TypeId::BOOLEAN: *reinterpret_cast<bool*>(dst) = val.get<bool>(); break;
        case TypeId::TIMESTAMP: *reinterpret_cast<int64_t*>(dst) = val.as_int64(); break;
        default: break;
    }

    bit_utils::set_bit(null_bitmap_, size_);
    size_++;
}

void Column::append_null() {
    if (size_ >= capacity_) {
        reserve(std::max(capacity_ * 2, static_cast<size_t>(64)));
    }
    bit_utils::clear_bit(null_bitmap_, size_);
    null_count_++;
    size_++;
}

void Column::add_string(std::string_view str) {
    if (size_ >= capacity_) {
        reserve(std::max(capacity_ * 2, static_cast<size_t>(64)));
    }
    uint32_t offset = static_cast<uint32_t>(string_heap_.size());
    uint32_t len = static_cast<uint32_t>(str.size());
    string_heap_.insert(string_heap_.end(), str.begin(), str.end());

    string_offsets_.push_back(offset);
    string_lengths_.push_back(len);

    uint32_t* dst = static_cast<uint32_t*>(data_) + size_;
    *dst = offset;

    bit_utils::set_bit(null_bitmap_, size_);
    size_++;
}

std::string_view Column::get_string(size_t row_idx) const {
    if (row_idx >= size_ || row_idx >= string_offsets_.size()) return "";
    uint32_t off = string_offsets_[row_idx];
    uint32_t len = string_lengths_[row_idx];
    return {string_heap_.data() + off, len};
}

ScalarValue Column::get_scalar(size_t row_idx) const {
    if (row_idx >= size_) return ScalarValue::null(type_);
    if (!bit_utils::get_bit(null_bitmap_, row_idx)) {
        return ScalarValue::null(type_);
    }

    if (type_.id == TypeId::STRING) {
        return ScalarValue::make_string(std::string(get_string(row_idx)));
    }

    const char* src = static_cast<const char*>(data_) + (row_idx * type_.width);
    switch (type_.id) {
        case TypeId::INT8:    return ScalarValue::make_int8(*reinterpret_cast<const int8_t*>(src));
        case TypeId::INT16:   return ScalarValue::make_int16(*reinterpret_cast<const int16_t*>(src));
        case TypeId::INT32:   return ScalarValue::make_int32(*reinterpret_cast<const int32_t*>(src));
        case TypeId::INT64:   return ScalarValue::make_int64(*reinterpret_cast<const int64_t*>(src));
        case TypeId::UINT8:   return ScalarValue::make_uint8(*reinterpret_cast<const uint8_t*>(src));
        case TypeId::UINT16:  return ScalarValue::make_uint16(*reinterpret_cast<const uint16_t*>(src));
        case TypeId::UINT32:  return ScalarValue::make_uint32(*reinterpret_cast<const uint32_t*>(src));
        case TypeId::UINT64:  return ScalarValue::make_uint64(*reinterpret_cast<const uint64_t*>(src));
        case TypeId::FLOAT32: return ScalarValue::make_float32(*reinterpret_cast<const float*>(src));
        case TypeId::FLOAT64: return ScalarValue::make_float64(*reinterpret_cast<const double*>(src));
        case TypeId::BOOLEAN: return ScalarValue::make_bool(*reinterpret_cast<const bool*>(src));
        case TypeId::TIMESTAMP: return ScalarValue::make_timestamp(*reinterpret_cast<const int64_t*>(src));
        default: return ScalarValue::null(type_);
    }
}

void Column::to_device(cudaStream_t stream) {
    if (on_device_) return;
    void* dev_data = cuda::GpuMemoryPool::global().allocate(capacity_ * type_.width);
    uint8_t* dev_bitmap = static_cast<uint8_t*>(cuda::GpuMemoryPool::global().allocate((capacity_ + 7) / 8));

    cudaMemcpyAsync(dev_data, data_, size_ * type_.width, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_bitmap, null_bitmap_, (size_ + 7) / 8, cudaMemcpyHostToDevice, stream);

    free(data_);
    free(null_bitmap_);
    data_ = dev_data;
    null_bitmap_ = dev_bitmap;
    on_device_ = true;
}

void Column::to_host(cudaStream_t stream) {
    if (!on_device_) return;
    void* host_data = nullptr;
    uint8_t* host_bitmap = nullptr;
    posix_memalign(&host_data, 128, std::max(capacity_ * type_.width, static_cast<size_t>(128)));
    posix_memalign(reinterpret_cast<void**>(&host_bitmap), 128, std::max((capacity_ + 7) / 8, static_cast<size_t>(128)));

    cudaMemcpyAsync(host_data, data_, size_ * type_.width, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(host_bitmap, null_bitmap_, (size_ + 7) / 8, cudaMemcpyDeviceToHost, stream);

    cuda::GpuMemoryPool::global().deallocate(data_, capacity_ * type_.width);
    cuda::GpuMemoryPool::global().deallocate(null_bitmap_, (capacity_ + 7) / 8);

    data_ = host_data;
    null_bitmap_ = host_bitmap;
    on_device_ = false;
}

std::shared_ptr<Column> Column::gather(const uint32_t* row_ids, size_t count, [[maybe_unused]] cudaStream_t stream) const {
    auto res = std::make_shared<Column>(type_, count, false);
    for (size_t i = 0; i < count; ++i) {
        uint32_t r = row_ids[i];
        res->append_scalar(get_scalar(r));
    }
    return res;
}

} // namespace onyx::storage
