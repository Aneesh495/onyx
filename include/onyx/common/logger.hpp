#pragma once

#include <string>
#include <string_view>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>

namespace onyx {

enum class LogLevel : uint8_t {
    TRACE = 0,
    DEBUG,
    INFO,
    WARN,
    ERROR,
    FATAL
};

class Logger {
public:
    static Logger& instance() {
        static Logger instance;
        return instance;
    }

    void set_level(LogLevel level) noexcept {
        level_ = level;
    }

    [[nodiscard]] LogLevel get_level() const noexcept {
        return level_;
    }

    void log(LogLevel level, std::string_view file, int line, const std::string& msg) {
        if (level < level_) return;

        auto now = std::chrono::system_clock::now();
        auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

        std::lock_guard<std::mutex> lock(mutex_);
        std::cerr << "[" << micros << "] [" << level_name(level) << "] ["
                  << file << ":" << line << "] " << msg << "\n";
    }

private:
    Logger() : level_(LogLevel::INFO) {}

    static const char* level_name(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO ";
            case LogLevel::WARN:  return "WARN ";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::FATAL: return "FATAL";
        }
        return "UNKNOWN";
    }

    LogLevel level_;
    std::mutex mutex_;
};

#define ONYX_LOG(level, msg) \
    do { \
        if (level >= ::onyx::Logger::instance().get_level()) { \
            std::ostringstream _oss; \
            _oss << msg; \
            ::onyx::Logger::instance().log(level, __FILE__, __LINE__, _oss.str()); \
        } \
    } while (0)

#define ONYX_LOG_TRACE(msg) ONYX_LOG(::onyx::LogLevel::TRACE, msg)
#define ONYX_LOG_DEBUG(msg) ONYX_LOG(::onyx::LogLevel::DEBUG, msg)
#define ONYX_LOG_INFO(msg)  ONYX_LOG(::onyx::LogLevel::INFO, msg)
#define ONYX_LOG_WARN(msg)  ONYX_LOG(::onyx::LogLevel::WARN, msg)
#define ONYX_LOG_ERROR(msg) ONYX_LOG(::onyx::LogLevel::ERROR, msg)
#define ONYX_LOG_FATAL(msg) ONYX_LOG(::onyx::LogLevel::FATAL, msg)

} // namespace onyx
