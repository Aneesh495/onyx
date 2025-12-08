#pragma once

#include "onyx/sql/token.hpp"
#include <string>
#include <string_view>
#include <vector>

namespace onyx::sql {

class Lexer {
public:
    explicit Lexer(std::string_view source);

    [[nodiscard]] Token next_token();
    [[nodiscard]] std::vector<Token> tokenize_all();

private:
    [[nodiscard]] char peek() const noexcept;
    [[nodiscard]] char peek_next() const noexcept;
    char advance() noexcept;
    void skip_whitespace();

    Token scan_identifier();
    Token scan_number();
    Token scan_string();

    std::string_view source_;
    size_t cursor_{0};
    uint32_t line_{1};
    uint32_t column_{1};
};

} // namespace onyx::sql
