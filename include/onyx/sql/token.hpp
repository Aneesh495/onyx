#pragma once

#include <string>
#include <string_view>
#include <cstdint>

namespace onyx::sql {

enum class TokenType : uint16_t {
    EndOfFile = 0,
    Identifier,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,

    // Keywords
    Select,
    From,
    Where,
    Group,
    By,
    Having,
    Order,
    Asc,
    Desc,
    Limit,
    Offset,
    Join,
    Inner,
    Left,
    On,
    As,
    And,
    Or,
    Not,
    Over,
    Partition,
    Rows,
    Between,
    Preceding,
    Current,
    Row,
    Like,
    In,
    Is,
    Null,
    True,
    False,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,

    // Punctuation
    Comma,
    Semicolon,
    Dot,
    LeftParen,
    RightParen
};

struct Token {
    TokenType type{TokenType::EndOfFile};
    std::string text;
    uint32_t line{1};
    uint32_t column{1};

    [[nodiscard]] std::string to_string() const;
};

std::string_view token_type_name(TokenType type);

} // namespace onyx::sql
