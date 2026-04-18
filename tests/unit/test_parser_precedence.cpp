#include "test_framework.hpp"
#include "onyx/sql/parser.hpp"

ONYX_TEST(SqlParserArithmeticPrecedence) {
    std::string sql = "SELECT a + b * c FROM t WHERE x > 10 AND y <= 20;";
    onyx::sql::Parser parser(sql);
    auto stmt = parser.parse_statement();
    ONYX_ASSERT(stmt != nullptr);
    ONYX_ASSERT_EQ(stmt->select_list.size(), 1);
    ONYX_ASSERT(stmt->where_clause != nullptr);
}

ONYX_TEST(SqlParserParenthesesNesting) {
    std::string sql = "SELECT (a + b) * (c - d) FROM t;";
    onyx::sql::Parser parser(sql);
    auto stmt = parser.parse_statement();
    ONYX_ASSERT(stmt != nullptr);
    ONYX_ASSERT_EQ(stmt->select_list.size(), 1);
}
