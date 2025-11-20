#pragma once

#include "onyx/common/status.hpp"
#include <stdexcept>
#include <string>

namespace onyx {

class OnyxException : public std::runtime_error {
public:
    explicit OnyxException(Status status)
        : std::runtime_error(status.to_string()), status_(std::move(status)) {}

    explicit OnyxException(StatusCode code, std::string message)
        : std::runtime_error(message), status_(code, std::move(message)) {}

    [[nodiscard]] const Status& status() const noexcept { return status_; }
    [[nodiscard]] StatusCode code() const noexcept { return status_.code(); }

private:
    Status status_;
};

class CudaKernelException : public OnyxException {
public:
    explicit CudaKernelException(std::string message)
        : OnyxException(StatusCode::CudaError, std::move(message)) {}
};

class ParserException : public OnyxException {
public:
    explicit ParserException(std::string message)
        : OnyxException(StatusCode::ParserError, std::move(message)) {}
};

class PlannerException : public OnyxException {
public:
    explicit PlannerException(std::string message)
        : OnyxException(StatusCode::PlannerError, std::move(message)) {}
};

class ExecutionException : public OnyxException {
public:
    explicit ExecutionException(std::string message)
        : OnyxException(StatusCode::ExecutionError, std::move(message)) {}
};

} // namespace onyx
