#include "onyx/common/types.hpp"
#include <sstream>

namespace onyx {

std::string DataType::to_string() const {
    switch (id) {
        case TypeId::INVALID:    return "INVALID";
        case TypeId::INT8:       return "INT8";
        case TypeId::INT16:      return "INT16";
        case TypeId::INT32:      return "INT32";
        case TypeId::INT64:      return "INT64";
        case TypeId::UINT8:      return "UINT8";
        case TypeId::UINT16:     return "UINT16";
        case TypeId::UINT32:     return "UINT32";
        case TypeId::UINT64:     return "UINT64";
        case TypeId::FLOAT32:    return "FLOAT32";
        case TypeId::FLOAT64:    return "FLOAT64";
        case TypeId::BOOLEAN:    return "BOOLEAN";
        case TypeId::STRING:     return "STRING";
        case TypeId::TIMESTAMP:  return "TIMESTAMP";
        case TypeId::DICTIONARY: return "DICTIONARY";
    }
    return "UNKNOWN";
}

int64_t ScalarValue::as_int64() const {
    if (is_null_) return 0;
    switch (type_.id) {
        case TypeId::INT8:    return std::get<int8_t>(storage_);
        case TypeId::INT16:   return std::get<int16_t>(storage_);
        case TypeId::INT32:   return std::get<int32_t>(storage_);
        case TypeId::INT64:   return std::get<int64_t>(storage_);
        case TypeId::UINT8:   return std::get<uint8_t>(storage_);
        case TypeId::UINT16:  return std::get<uint16_t>(storage_);
        case TypeId::UINT32:  return std::get<uint32_t>(storage_);
        case TypeId::UINT64:  return static_cast<int64_t>(std::get<uint64_t>(storage_));
        case TypeId::BOOLEAN: return std::get<bool>(storage_) ? 1 : 0;
        case TypeId::TIMESTAMP: return std::get<int64_t>(storage_);
        default: return 0;
    }
}

double ScalarValue::as_double() const {
    if (is_null_) return 0.0;
    switch (type_.id) {
        case TypeId::FLOAT32: return std::get<float>(storage_);
        case TypeId::FLOAT64: return std::get<double>(storage_);
        default: return static_cast<double>(as_int64());
    }
}

std::string ScalarValue::to_string() const {
    if (is_null_) return "NULL";
    switch (type_.id) {
        case TypeId::INT8:       return std::to_string(std::get<int8_t>(storage_));
        case TypeId::INT16:      return std::to_string(std::get<int16_t>(storage_));
        case TypeId::INT32:      return std::to_string(std::get<int32_t>(storage_));
        case TypeId::INT64:      return std::to_string(std::get<int64_t>(storage_));
        case TypeId::UINT8:      return std::to_string(std::get<uint8_t>(storage_));
        case TypeId::UINT16:     return std::to_string(std::get<uint16_t>(storage_));
        case TypeId::UINT32:     return std::to_string(std::get<uint32_t>(storage_));
        case TypeId::UINT64:     return std::to_string(std::get<uint64_t>(storage_));
        case TypeId::FLOAT32:    return std::to_string(std::get<float>(storage_));
        case TypeId::FLOAT64:    return std::to_string(std::get<double>(storage_));
        case TypeId::BOOLEAN:    return std::get<bool>(storage_) ? "true" : "false";
        case TypeId::STRING:     return std::get<std::string>(storage_);
        case TypeId::TIMESTAMP:  return std::to_string(std::get<int64_t>(storage_)) + "us";
        default:                 return "<unknown>";
    }
}

bool ScalarValue::operator==(const ScalarValue& other) const {
    if (is_null_ && other.is_null_) return true;
    if (is_null_ || other.is_null_) return false;
    if (type_.id != other.type_.id) {
        if (type_.is_numeric() && other.type_.is_numeric()) {
            return as_double() == other.as_double();
        }
        return false;
    }
    return storage_ == other.storage_;
}

bool ScalarValue::operator<(const ScalarValue& other) const {
    if (is_null_ && other.is_null_) return false;
    if (is_null_) return true;
    if (other.is_null_) return false;
    if (type_.is_numeric() && other.type_.is_numeric()) {
        return as_double() < other.as_double();
    }
    if (type_.id == TypeId::STRING && other.type_.id == TypeId::STRING) {
        return std::get<std::string>(storage_) < std::get<std::string>(other.storage_);
    }
    return false;
}

std::ostream& operator<<(std::ostream& os, const DataType& dt) {
    return os << dt.to_string();
}

std::ostream& operator<<(std::ostream& os, const ScalarValue& val) {
    return os << val.to_string();
}

} // namespace onyx
