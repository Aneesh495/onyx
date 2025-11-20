#include "onyx/common/status.hpp"
#include <sstream>

namespace onyx {

std::string Status::to_string() const {
    if (is_ok()) {
        return "OK";
    }

    std::string prefix;
    switch (code_) {
        case StatusCode::InvalidArgument: prefix = "InvalidArgument: "; break;
        case StatusCode::OutOfMemory:     prefix = "OutOfMemory: "; break;
        case StatusCode::CudaError:       prefix = "CudaError: "; break;
        case StatusCode::SchemaMismatch:  prefix = "SchemaMismatch: "; break;
        case StatusCode::TableNotFound:   prefix = "TableNotFound: "; break;
        case StatusCode::ColumnNotFound:  prefix = "ColumnNotFound: "; break;
        case StatusCode::ParserError:     prefix = "ParserError: "; break;
        case StatusCode::PlannerError:    prefix = "PlannerError: "; break;
        case StatusCode::ExecutionError:  prefix = "ExecutionError: "; break;
        case StatusCode::NotImplemented:  prefix = "NotImplemented: "; break;
        case StatusCode::Timeout:         prefix = "Timeout: "; break;
        case StatusCode::ConcurrencyError:prefix = "ConcurrencyError: "; break;
        case StatusCode::IoError:         prefix = "IoError: "; break;
        default:                          prefix = "Unknown: "; break;
    }

    return prefix + message_;
}

std::ostream& operator<<(std::ostream& os, const Status& status) {
    return os << status.to_string();
}

} // namespace onyx
