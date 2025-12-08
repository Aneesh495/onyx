#include "onyx/sql/parser.hpp"
#include "onyx/common/error.hpp"
#include <stdexcept>

namespace onyx::sql {

Parser::Parser(std::string_view sql) {
    Lexer lexer(sql);
    tokens_ = lexer.tokenize_all();
}

const Token& Parser::peek() const {
    if (cursor_ >= tokens_.size()) return tokens_.back();
    return tokens_[cursor_];
}

Token Parser::advance() {
    if (cursor_ >= tokens_.size()) return tokens_.back();
    return tokens_[cursor_++];
}

bool Parser::check(TokenType type) const {
    return peek().type == type;
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

Token Parser::consume(TokenType type, std::string_view error_msg) {
    if (check(type)) {
        return advance();
    }
    throw ParserException(std::string(error_msg) + " at token: " + peek().to_string());
}

int Parser::get_precedence(TokenType type) const {
    switch (type) {
        case TokenType::Or: return 1;
        case TokenType::And: return 2;
        case TokenType::Equal:
        case TokenType::NotEqual:
        case TokenType::LessThan:
        case TokenType::LessThanOrEqual:
        case TokenType::GreaterThan:
        case TokenType::GreaterThanOrEqual:
        case TokenType::Like: return 3;
        case TokenType::Plus:
        case TokenType::Minus: return 4;
        case TokenType::Star:
        case TokenType::Slash: return 5;
        default: return 0;
    }
}

BinaryOp Parser::token_to_binary_op(TokenType type) const {
    switch (type) {
        case TokenType::Plus:               return BinaryOp::Add;
        case TokenType::Minus:              return BinaryOp::Subtract;
        case TokenType::Star:               return BinaryOp::Multiply;
        case TokenType::Slash:              return BinaryOp::Divide;
        case TokenType::Equal:              return BinaryOp::Equal;
        case TokenType::NotEqual:           return BinaryOp::NotEqual;
        case TokenType::LessThan:           return BinaryOp::LessThan;
        case TokenType::LessThanOrEqual:    return BinaryOp::LessThanOrEqual;
        case TokenType::GreaterThan:        return BinaryOp::GreaterThan;
        case TokenType::GreaterThanOrEqual: return BinaryOp::GreaterThanOrEqual;
        case TokenType::And:                return BinaryOp::And;
        case TokenType::Or:                 return BinaryOp::Or;
        case TokenType::Like:               return BinaryOp::Like;
        default: throw ParserException("Not a binary operator");
    }
}

std::shared_ptr<Expr> Parser::parse_primary() {
    if (match(TokenType::IntegerLiteral)) {
        int64_t val = std::stoll(tokens_[cursor_ - 1].text);
        return std::make_shared<LiteralExpr>(ScalarValue::make_int64(val));
    }
    if (match(TokenType::FloatLiteral)) {
        double val = std::stod(tokens_[cursor_ - 1].text);
        return std::make_shared<LiteralExpr>(ScalarValue::make_float64(val));
    }
    if (match(TokenType::StringLiteral)) {
        return std::make_shared<LiteralExpr>(ScalarValue::make_string(tokens_[cursor_ - 1].text));
    }
    if (match(TokenType::True)) {
        return std::make_shared<LiteralExpr>(ScalarValue::make_bool(true));
    }
    if (match(TokenType::False)) {
        return std::make_shared<LiteralExpr>(ScalarValue::make_bool(false));
    }
    if (match(TokenType::Null)) {
        return std::make_shared<LiteralExpr>(ScalarValue::null(DataType(TypeId::INT64)));
    }
    if (match(TokenType::Star)) {
        return std::make_shared<ColumnRefExpr>("*");
    }

    if (match(TokenType::Identifier)) {
        std::string name = tokens_[cursor_ - 1].text;
        if (match(TokenType::Dot)) {
            Token col_tok = consume(TokenType::Identifier, "Expected column name after dot");
            return std::make_shared<ColumnRefExpr>(col_tok.text, name);
        }
        if (check(TokenType::LeftParen)) {
            auto func = parse_function_call(name);
            if (check(TokenType::Over)) {
                return parse_window_clause(func);
            }
            return func;
        }
        return std::make_shared<ColumnRefExpr>(name);
    }

    if (match(TokenType::LeftParen)) {
        auto expr = parse_expression(0);
        consume(TokenType::RightParen, "Expected ')' after expression");
        return expr;
    }

    throw ParserException("Unexpected token in primary expression: " + peek().to_string());
}

std::shared_ptr<FunctionCallExpr> Parser::parse_function_call(std::string func_name) {
    consume(TokenType::LeftParen, "Expected '(' after function name");
    std::vector<std::shared_ptr<Expr>> args;
    if (!check(TokenType::RightParen)) {
        do {
            args.push_back(parse_expression(0));
        } while (match(TokenType::Comma));
    }
    consume(TokenType::RightParen, "Expected ')' after function arguments");
    return std::make_shared<FunctionCallExpr>(std::move(func_name), std::move(args));
}

std::shared_ptr<WindowExpr> Parser::parse_window_clause(std::shared_ptr<FunctionCallExpr> func_call) {
    consume(TokenType::Over, "Expected 'OVER' keyword");
    consume(TokenType::LeftParen, "Expected '(' after OVER");

    std::vector<std::shared_ptr<Expr>> partition_by;
    if (match(TokenType::Partition)) {
        consume(TokenType::By, "Expected 'BY' after PARTITION");
        do {
            partition_by.push_back(parse_expression(0));
        } while (match(TokenType::Comma));
    }

    std::vector<std::pair<std::shared_ptr<Expr>, bool>> order_by;
    if (match(TokenType::Order)) {
        consume(TokenType::By, "Expected 'BY' after ORDER");
        do {
            auto expr = parse_expression(0);
            bool asc = true;
            if (match(TokenType::Asc)) asc = true;
            else if (match(TokenType::Desc)) asc = false;
            order_by.emplace_back(expr, asc);
        } while (match(TokenType::Comma));
    }

    uint32_t preceding = 0;
    if (match(TokenType::Rows)) {
        consume(TokenType::Between, "Expected 'BETWEEN' after ROWS");
        Token num = consume(TokenType::IntegerLiteral, "Expected integer preceding rows");
        preceding = static_cast<uint32_t>(std::stoul(num.text));
        consume(TokenType::Preceding, "Expected 'PRECEDING'");
        consume(TokenType::And, "Expected 'AND'");
        consume(TokenType::Current, "Expected 'CURRENT'");
        consume(TokenType::Row, "Expected 'ROW'");
    }

    consume(TokenType::RightParen, "Expected ')' to close OVER clause");
    return std::make_shared<WindowExpr>(std::move(func_call), std::move(partition_by), std::move(order_by), preceding);
}

std::shared_ptr<Expr> Parser::parse_expression(int min_precedence) {
    auto left = parse_primary();

    while (true) {
        TokenType op_type = peek().type;
        int prec = get_precedence(op_type);
        if (prec <= min_precedence) {
            break;
        }

        advance(); // consume binary operator
        auto right = parse_expression(prec);
        left = std::make_shared<BinaryExpr>(left, token_to_binary_op(op_type), right);
    }

    return left;
}

std::shared_ptr<SelectStatement> Parser::parse_statement() {
    auto stmt = std::make_shared<SelectStatement>();

    consume(TokenType::Select, "Expected 'SELECT' at start of statement");

    // Select items
    do {
        auto expr = parse_expression(0);
        std::string alias;
        if (match(TokenType::As)) {
            Token t = consume(TokenType::Identifier, "Expected alias after AS");
            alias = t.text;
        } else if (check(TokenType::Identifier) && !check(TokenType::From)) {
            Token t = advance();
            alias = t.text;
        }
        stmt->select_list.push_back(SelectItem{.expr = expr, .alias = alias});
    } while (match(TokenType::Comma));

    // FROM clause
    if (match(TokenType::From)) {
        Token table_tok = consume(TokenType::Identifier, "Expected table name after FROM");
        stmt->from_table = table_tok.text;
        if (check(TokenType::Identifier) && !check(TokenType::Where) &&
            !check(TokenType::Group) && !check(TokenType::Order) && !check(TokenType::Join) &&
            !check(TokenType::Inner) && !check(TokenType::Left) && !check(TokenType::Limit)) {
            stmt->from_alias = advance().text;
        }
    }

    // JOIN clauses
    while (check(TokenType::Join) || check(TokenType::Inner) || check(TokenType::Left)) {
        bool is_left = false;
        if (match(TokenType::Left)) {
            consume(TokenType::Join, "Expected 'JOIN' after LEFT");
            is_left = true;
        } else {
            match(TokenType::Inner);
            consume(TokenType::Join, "Expected 'JOIN'");
        }

        Token join_table = consume(TokenType::Identifier, "Expected joined table name");
        std::string join_alias;
        if (check(TokenType::Identifier) && !check(TokenType::On)) {
            join_alias = advance().text;
        }

        consume(TokenType::On, "Expected 'ON' after JOIN table");
        auto on_cond = parse_expression(0);

        stmt->joins.push_back(JoinClause{
            .table_name = join_table.text,
            .table_alias = join_alias,
            .on_condition = on_cond,
            .is_left = is_left
        });
    }

    // WHERE clause
    if (match(TokenType::Where)) {
        stmt->where_clause = parse_expression(0);
    }

    // GROUP BY clause
    if (match(TokenType::Group)) {
        consume(TokenType::By, "Expected 'BY' after GROUP");
        do {
            stmt->group_by.push_back(parse_expression(0));
        } while (match(TokenType::Comma));
    }

    // HAVING clause
    if (match(TokenType::Having)) {
        stmt->having_clause = parse_expression(0);
    }

    // ORDER BY clause
    if (match(TokenType::Order)) {
        consume(TokenType::By, "Expected 'BY' after ORDER");
        do {
            auto expr = parse_expression(0);
            bool asc = true;
            if (match(TokenType::Asc)) asc = true;
            else if (match(TokenType::Desc)) asc = false;
            stmt->order_by.push_back(OrderByItem{.expr = expr, .ascending = asc});
        } while (match(TokenType::Comma));
    }

    // LIMIT clause
    if (match(TokenType::Limit)) {
        Token num = consume(TokenType::IntegerLiteral, "Expected integer for LIMIT");
        stmt->limit_count = std::stoll(num.text);
    }

    // OFFSET clause
    if (match(TokenType::Offset)) {
        Token num = consume(TokenType::IntegerLiteral, "Expected integer for OFFSET");
        stmt->offset_count = std::stoll(num.text);
    }

    match(TokenType::Semicolon);
    return stmt;
}

} // namespace onyx::sql
