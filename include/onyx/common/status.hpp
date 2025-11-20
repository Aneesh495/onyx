#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <ostream>

namespace onyx {

enum class StatusCode : uint32_t {
    Ok = 0,
    InvalidArgument,
    OutOfMemory,
    CudaError,
    SchemaMismatch,
    TableNotFound,
    ColumnNotFound,
    ParserError,
    PlannerError,
    ExecutionError,
    NotImplemented,
    Timeout,
    ConcurrencyError,
    IoError,
    Unknown
};

class [[nodiscard]] Status {
public:
    Status() noexcept : code_(StatusCode::Ok) {}
    Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

    static Status ok() noexcept { return Status(); }
    static Status invalid_argument(std::string message) { return {StatusCode::InvalidArgument, std::move(message)}; }
    static Status out_of_memory(std::string message) { return {StatusCode::OutOfMemory, std::move(message)}; }
    static Status cuda_error(std::string message) { return {StatusCode::CudaError, std::move(message)}; }
    static Status schema_mismatch(std::string message) { return {StatusCode::SchemaMismatch, std::move(message)}; }
    static Status table_not_found(std::string message) { return {StatusCode::TableNotFound, std::move(message)}; }
    static Status column_not_found(std::string message) { return {StatusCode::ColumnNotFound, std::move(message)}; }
    static Status parser_error(std::string message) { return {StatusCode::ParserError, std::move(message)}; }
    static Status planner_error(std::string message) { return {StatusCode::PlannerError, std::move(message)}; }
    static Status execution_error(std::string message) { return {StatusCode::ExecutionError, std::move(message)}; }
    static Status not_implemented(std::string message) { return {StatusCode::NotImplemented, std::move(message)}; }
    static Status timeout(std::string message) { return {StatusCode::Timeout, std::move(message)}; }
    static Status io_error(std::string message) { return {StatusCode::IoError, std::move(message)}; }

    [[nodiscard]] bool is_ok() const noexcept { return code_ == StatusCode::Ok; }
    [[nodiscard]] bool is_error() const noexcept { return !is_ok(); }
    [[nodiscard]] StatusCode code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }

    [[nodiscard]] std::string to_string() const;

    explicit operator bool() const noexcept { return is_ok(); }

private:
    StatusCode code_;
    std::string message_;
};

#define ONYX_RETURN_IF_ERROR(expr)       \
    do {                                 \
        auto _status = (expr);           \
        if (!_status.is_ok()) {          \
            return _status;              \
        }                                \
    } while (0)

std::ostream& operator<<(std::ostream& os, const Status& status);

} // namespace onyx
