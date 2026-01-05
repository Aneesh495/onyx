#include "test_framework.hpp"
#include "onyx/sql/parser.hpp"
#include <stdexcept>


ONYX_TEST(SqlParserSelectWhereLimit) {
    std::string sql = "SELECT a, b * 2 AS c FROM my_table WHERE a > 10 ORDER BY b DESC LIMIT 5;";
    onyx::sql::Parser parser(sql);
    auto stmt = parser.parse_statement();

    ONYX_ASSERT(stmt != nullptr);
    ONYX_ASSERT_EQ(stmt->from_table, "my_table");
    ONYX_ASSERT_EQ(stmt->select_list.size(), 2);
    ONYX_ASSERT_EQ(stmt->select_list[1].alias, "c");
    ONYX_ASSERT(stmt->where_clause != nullptr);
    ONYX_ASSERT_EQ(stmt->order_by.size(), 1);
    ONYX_ASSERT_EQ(stmt->order_by[0].ascending, false);
    ONYX_ASSERT_EQ(stmt->limit_count, 5);
}

ONYX_TEST(SqlParserJoinAndWindow) {
    std::string sql = "SELECT t1.id, avg(t1.val) OVER (PARTITION BY t1.cat ORDER BY t1.ts ASC ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) FROM t1 INNER JOIN t2 ON t1.id = t2.ref_id;";
    onyx::sql::Parser parser(sql);
    auto stmt = parser.parse_statement();

    ONYX_ASSERT(stmt != nullptr);
    ONYX_ASSERT_EQ(stmt->from_table, "t1");
    ONYX_ASSERT_EQ(stmt->joins.size(), 1);
    ONYX_ASSERT_EQ(stmt->joins[0].table_name, "t2");
    ONYX_ASSERT(stmt->joins[0].on_condition != nullptr);
    ONYX_ASSERT_EQ(stmt->select_list.size(), 2);
}
