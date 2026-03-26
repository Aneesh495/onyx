#pragma once

#include <cstdint>
#include <string>

namespace onyx::common {

class DateUtils {
public:
    // Parse ISO-8601 date string (YYYY-MM-DD) into epoch days
    static int32_t parse_date(std::string_view date_str);

    // Format epoch days into ISO-8601 date string (YYYY-MM-DD)
    static std::string format_date(int32_t epoch_days);

    // Parse ISO-8601 timestamp string (YYYY-MM-DDTHH:MM:SS) into microseconds since epoch
    static int64_t parse_timestamp(std::string_view ts_str);

    // Format microseconds since epoch into ISO-8601 timestamp string
    static std::string format_timestamp(int64_t epoch_micros);
};

} // namespace onyx::common
