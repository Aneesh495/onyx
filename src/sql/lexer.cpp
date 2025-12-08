#include "onyx/sql/lexer.hpp"
#include <cctype>
#include <unordered_map>

namespace onyx::sql {

static const std::unordered_map<std::string, TokenType> KEYWORDS = {
    {"select", TokenType::Select},
    {"from", TokenType::From},
    {"where", TokenType::Where},
    {"group", TokenType::Group},
    {"by", TokenType::By},
    {"having", TokenType::Having},
    {"order", TokenType::Order},
    {"asc", TokenType::Asc},
    {"desc", TokenType::Desc},
    {"limit", TokenType::Limit},
    {"offset", TokenType::Offset},
    {"join", TokenType::Join},
    {"inner", TokenType::Inner},
    {"left", TokenType::Left},
    {"on", TokenType::On},
    {"as", TokenType::As},
    {"and", TokenType::And},
    {"or", TokenType::Or},
    {"not", TokenType::Not},
    {"over", TokenType::Over},
    {"partition", TokenType::Partition},
    {"rows", TokenType::Rows},
    {"between", TokenType::Between},
    {"preceding", TokenType::Preceding},
    {"current", TokenType::Current},
    {"row", TokenType::Row},
    {"like", TokenType::Like},
    {"in", TokenType::In},
    {"is", TokenType::Is},
    {"null", TokenType::Null},
    {"true", TokenType::True},
    {"false", TokenType::False}
};

Lexer::Lexer(std::string_view source) : source_(source) {}

char Lexer::peek() const noexcept {
    if (cursor_ >= source_.size()) return '\0';
    return source_[cursor_];
}

char Lexer::peek_next() const noexcept {
    if (cursor_ + 1 >= source_.size()) return '\0';
    return source_[cursor_ + 1];
}

char Lexer::advance() noexcept {
    if (cursor_ >= source_.size()) return '\0';
    char c = source_[cursor_++];
    if (c == '\n') {
        line_++;
        column_ = 1;
    } else {
        column_++;
    }
    return c;
}

void Lexer::skip_whitespace() {
    while (cursor_ < source_.size()) {
        char c = peek();
        if (std::isspace(c)) {
            advance();
        } else if (c == '-' && peek_next() == '-') {
            // Line comment: skip until newline
            while (cursor_ < source_.size() && peek() != '\n') {
                advance();
            }
        } else {
            break;
        }
    }
}

Token Lexer::scan_identifier() {
    uint32_t start_col = column_;
    size_t start_pos = cursor_;

    while (cursor_ < source_.size() && (std::isalnum(peek()) || peek() == '_')) {
        advance();
    }

    std::string text(source_.substr(start_pos, cursor_ - start_pos));
    std::string lower_text = text;
    for (char& c : lower_text) c = std::tolower(c);

    TokenType type = TokenType::Identifier;
    auto it = KEYWORDS.find(lower_text);
    if (it != KEYWORDS.end()) {
        type = it->second;
    }

    return Token{
        .type = type,
        .text = std::move(text),
        .line = line_,
        .column = start_col
    };
}

Token Lexer::scan_number() {
    uint32_t start_col = column_;
    size_t start_pos = cursor_;
    bool is_float = false;

    while (cursor_ < source_.size() && std::isdigit(peek())) {
        advance();
    }

    if (peek() == '.' && std::isdigit(peek_next())) {
        is_float = true;
        advance(); // consume '.'
        while (cursor_ < source_.size() && std::isdigit(peek())) {
            advance();
        }
    }

    return Token{
        .type = is_float ? TokenType::FloatLiteral : TokenType::IntegerLiteral,
        .text = std::string(source_.substr(start_pos, cursor_ - start_pos)),
        .line = line_,
        .column = start_col
    };
}

Token Lexer::scan_string() {
    uint32_t start_col = column_;
    advance(); // consume opening quote
    size_t start_pos = cursor_;

    while (cursor_ < source_.size() && peek() != '\'') {
        advance();
    }

    std::string text(source_.substr(start_pos, cursor_ - start_pos));
    if (peek() == '\'') {
        advance(); // consume closing quote
    }

    return Token{
        .type = TokenType::StringLiteral,
        .text = std::move(text),
        .line = line_,
        .column = start_col
    };
}

Token Lexer::next_token() {
    skip_whitespace();

    if (cursor_ >= source_.size()) {
        return Token{.type = TokenType::EndOfFile, .text = "", .line = line_, .column = column_};
    }

    uint32_t cur_col = column_;
    char c = peek();

    if (std::isalpha(c) || c == '_') {
        return scan_identifier();
    }
    if (std::isdigit(c)) {
        return scan_number();
    }
    if (c == '\'') {
        return scan_string();
    }

    advance();
    switch (c) {
        case '+': return Token{TokenType::Plus, "+", line_, cur_col};
        case '-': return Token{TokenType::Minus, "-", line_, cur_col};
        case '*': return Token{TokenType::Star, "*", line_, cur_col};
        case '/': return Token{TokenType::Slash, "/", line_, cur_col};
        case ',': return Token{TokenType::Comma, ",", line_, cur_col};
        case ';': return Token{TokenType::Semicolon, ";", line_, cur_col};
        case '.': return Token{TokenType::Dot, ".", line_, cur_col};
        case '(': return Token{TokenType::LeftParen, "(", line_, cur_col};
        case ')': return Token{TokenType::RightParen, ")", line_, cur_col};
        case '=': return Token{TokenType::Equal, "=", line_, cur_col};
        case '<':
            if (peek() == '=') {
                advance();
                return Token{TokenType::LessThanOrEqual, "<=", line_, cur_col};
            }
            if (peek() == '>') {
                advance();
                return Token{TokenType::NotEqual, "<>", line_, cur_col};
            }
            return Token{TokenType::LessThan, "<", line_, cur_col};
        case '>':
            if (peek() == '=') {
                advance();
                return Token{TokenType::GreaterThanOrEqual, ">=", line_, cur_col};
            }
            return Token{TokenType::GreaterThan, ">", line_, cur_col};
        case '!':
            if (peek() == '=') {
                advance();
                return Token{TokenType::NotEqual, "!=", line_, cur_col};
            }
            break;
    }

    return Token{TokenType::EndOfFile, "", line_, cur_col};
}

std::vector<Token> Lexer::tokenize_all() {
    std::vector<Token> tokens;
    while (true) {
        Token tok = next_token();
        tokens.push_back(tok);
        if (tok.type == TokenType::EndOfFile) break;
    }
    return tokens;
}

} // namespace onyx::sql
