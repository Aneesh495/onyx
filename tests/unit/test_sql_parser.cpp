#include "onyx/sql/parser.hpp"
#include <stdexcept>

extern void register_test(std::string name, std::function<void()> func);

#define ONYX_TEST(name) \
    static void test_##name(); \
    static int dummy_##name = (register_test(#name, test_##name), 0); \
    static void test_##name()

#define ONYX_ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while (0)
#define ONYX_ASSERT_EQ(a, b) do { if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b); } while (0)

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
