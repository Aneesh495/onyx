#include "test_framework.hpp"
#include "onyx/common/date_utils.hpp"

ONYX_TEST(DateUtilsIsoParsingAndFormatting) {
    std::string date_str = "2024-03-15";
    int32_t days = onyx::common::DateUtils::parse_date(date_str);
    ONYX_ASSERT(days > 0);
    std::string formatted = onyx::common::DateUtils::format_date(days);
    ONYX_ASSERT_EQ(formatted, date_str);
}

ONYX_TEST(TimestampUtilsIsoParsingAndFormatting) {
    std::string ts_str = "2024-03-15T10:30:00Z";
    int64_t micros = onyx::common::DateUtils::parse_timestamp(ts_str);
    ONYX_ASSERT(micros > 0);
    std::string formatted = onyx::common::DateUtils::format_timestamp(micros);
    ONYX_ASSERT_EQ(formatted, ts_str);
}
