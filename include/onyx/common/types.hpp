#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <variant>
#include <optional>
#include <format>
#include <ostream>

namespace onyx {

// Hardware alignment constants
constexpr size_t WARP_SIZE = 32;
constexpr size_t WARP_SIZE_LOG2 = 5;
constexpr size_t CACHE_LINE_BYTES = 128;
constexpr size_t GPU_PAGE_SIZE = 4096;
constexpr size_t DEFAULT_BLOCK_SIZE = 256;
constexpr size_t MAX_BLOCK_SIZE = 1024;
constexpr size_t DEFAULT_VECTOR_CHUNK_SIZE = 65536;

enum class TypeId : uint8_t {
    INVALID = 0,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    FLOAT32,
    FLOAT64,
    BOOLEAN,
    STRING,
    TIMESTAMP,
    DICTIONARY
};

struct DataType {
    TypeId id{TypeId::INVALID};
    uint32_t width{0}; // Width in bytes for fixed width types

    constexpr DataType() = default;
    constexpr explicit DataType(TypeId id_) : id(id_), width(type_width(id_)) {}

    static constexpr uint32_t type_width(TypeId t) {
        switch (t) {
            case TypeId::INT8:
            case TypeId::UINT8:
            case TypeId::BOOLEAN:
                return 1;
            case TypeId::INT16:
            case TypeId::UINT16:
                return 2;
            case TypeId::INT32:
            case TypeId::UINT32:
            case TypeId::FLOAT32:
                return 4;
            case TypeId::INT64:
            case TypeId::UINT64:
            case TypeId::FLOAT64:
            case TypeId::TIMESTAMP:
                return 8;
            case TypeId::STRING:
            case TypeId::DICTIONARY:
                return 4; // 32-bit offset or dictionary code
            default:
                return 0;
        }
    }

    [[nodiscard]] constexpr bool is_integral() const noexcept {
        return id == TypeId::INT8 || id == TypeId::INT16 || id == TypeId::INT32 ||
               id == TypeId::INT64 || id == TypeId::UINT8 || id == TypeId::UINT16 ||
               id == TypeId::UINT32 || id == TypeId::UINT64;
    }

    [[nodiscard]] constexpr bool is_floating_point() const noexcept {
        return id == TypeId::FLOAT32 || id == TypeId::FLOAT64;
    }

    [[nodiscard]] constexpr bool is_numeric() const noexcept {
        return is_integral() || is_floating_point();
    }

    [[nodiscard]] constexpr bool is_fixed_width() const noexcept {
        return id != TypeId::STRING;
    }

    [[nodiscard]] std::string to_string() const;

    constexpr bool operator==(const DataType& other) const = default;
};

// Scalar value variant representation for predicates and literals
class ScalarValue {
public:
    using Storage = std::variant<
        std::monostate,
        int8_t,
        int16_t,
        int32_t,
        int64_t,
        uint8_t,
        uint16_t,
        uint32_t,
        uint64_t,
        float,
        double,
        bool,
        std::string,
        int64_t // Timestamp in microseconds
    >;

    ScalarValue() : type_(TypeId::INVALID), is_null_(true), storage_(std::monostate{}) {}

    static ScalarValue null(DataType type) {
        ScalarValue val;
        val.type_ = type;
        val.is_null_ = true;
        return val;
    }

    static ScalarValue make_int8(int8_t v) { return {DataType(TypeId::INT8), v}; }
    static ScalarValue make_int16(int16_t v) { return {DataType(TypeId::INT16), v}; }
    static ScalarValue make_int32(int32_t v) { return {DataType(TypeId::INT32), v}; }
    static ScalarValue make_int64(int64_t v) { return {DataType(TypeId::INT64), v}; }
    static ScalarValue make_uint8(uint8_t v) { return {DataType(TypeId::UINT8), v}; }
    static ScalarValue make_uint16(uint16_t v) { return {DataType(TypeId::UINT16), v}; }
    static ScalarValue make_uint32(uint32_t v) { return {DataType(TypeId::UINT32), v}; }
    static ScalarValue make_uint64(uint64_t v) { return {DataType(TypeId::UINT64), v}; }
    static ScalarValue make_float32(float v) { return {DataType(TypeId::FLOAT32), v}; }
    static ScalarValue make_float64(double v) { return {DataType(TypeId::FLOAT64), v}; }
    static ScalarValue make_bool(bool v) { return {DataType(TypeId::BOOLEAN), v}; }
    static ScalarValue make_string(std::string v) { return {DataType(TypeId::STRING), std::move(v)}; }
    static ScalarValue make_timestamp(int64_t v) { return {DataType(TypeId::TIMESTAMP), v}; }

    [[nodiscard]] DataType type() const noexcept { return type_; }
    [[nodiscard]] bool is_null() const noexcept { return is_null_; }

    template <typename T>
    [[nodiscard]] const T& get() const {
        return std::get<T>(storage_);
    }

    [[nodiscard]] int64_t as_int64() const;
    [[nodiscard]] double as_double() const;
    [[nodiscard]] std::string to_string() const;

    bool operator==(const ScalarValue& other) const;
    bool operator<(const ScalarValue& other) const;

private:
    template <typename T>
    ScalarValue(DataType type, T val)
        : type_(type), is_null_(false), storage_(std::move(val)) {}

    DataType type_;
    bool is_null_{true};
    Storage storage_;
};

std::ostream& operator<<(std::ostream& os, const DataType& dt);
std::ostream& operator<<(std::ostream& os, const ScalarValue& val);

} // namespace onyx
