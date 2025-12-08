#pragma once

#include "onyx/sql/lexer.hpp"
#include "onyx/sql/ast.hpp"
#include <memory>
#include <string_view>

namespace onyx::sql {

class Parser {
public:
    explicit Parser(std::string_view sql);

    [[nodiscard]] std::shared_ptr<SelectStatement> parse_statement();
    [[nodiscard]] std::shared_ptr<Expr> parse_expression(int min_precedence = 0);

private:
    [[nodiscard]] const Token& peek() const;
    Token advance();
    bool match(TokenType type);
    bool check(TokenType type) const;
    Token consume(TokenType type, std::string_view error_msg);

    std::shared_ptr<Expr> parse_primary();
    std::shared_ptr<Expr> parse_unary();
    std::shared_ptr<FunctionCallExpr> parse_function_call(std::string func_name);
    std::shared_ptr<WindowExpr> parse_window_clause(std::shared_ptr<FunctionCallExpr> func_call);

    int get_precedence(TokenType type) const;
    BinaryOp token_to_binary_op(TokenType type) const;

    std::vector<Token> tokens_;
    size_t cursor_{0};
};

} // namespace onyx::sql
