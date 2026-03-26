#include "onyx/common/date_utils.hpp"
#include <chrono>
#include <sstream>
#include <iomanip>

namespace onyx::common {

int32_t DateUtils::parse_date(std::string_view date_str) {
    if (date_str.size() < 10) return 0;
    int year = std::stoi(std::string(date_str.substr(0, 4)));
    int month = std::stoi(std::string(date_str.substr(5, 2)));
    int day = std::stoi(std::string(date_str.substr(8, 2)));

    std::tm tm = {};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_isdst = 0;

    std::time_t t = timegm(&tm);
    return static_cast<int32_t>(t / 86400);
}

std::string DateUtils::format_date(int32_t epoch_days) {
    std::time_t t = static_cast<std::time_t>(epoch_days) * 86400;
    std::tm* tm = std::gmtime(&t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", tm);
    return std::string(buf);
}

int64_t DateUtils::parse_timestamp(std::string_view ts_str) {
    if (ts_str.size() < 19) return 0;
    int year = std::stoi(std::string(ts_str.substr(0, 4)));
    int month = std::stoi(std::string(ts_str.substr(5, 2)));
    int day = std::stoi(std::string(ts_str.substr(8, 2)));
    int hour = std::stoi(std::string(ts_str.substr(11, 2)));
    int min = std::stoi(std::string(ts_str.substr(14, 2)));
    int sec = std::stoi(std::string(ts_str.substr(17, 2)));

    std::tm tm = {};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = min;
    tm.tm_sec = sec;
    tm.tm_isdst = 0;

    std::time_t t = timegm(&tm);
    return static_cast<int64_t>(t) * 1000000LL;
}

std::string DateUtils::format_timestamp(int64_t epoch_micros) {
    std::time_t sec = epoch_micros / 1000000LL;
    std::tm* tm = std::gmtime(&sec);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", tm);
    return std::string(buf);
}

} // namespace onyx::common
