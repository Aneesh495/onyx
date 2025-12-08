#include "onyx/sql/token.hpp"
#include <sstream>

namespace onyx::sql {

std::string_view token_type_name(TokenType type) {
    switch (type) {
        case TokenType::EndOfFile:          return "EOF";
        case TokenType::Identifier:         return "Identifier";
        case TokenType::IntegerLiteral:     return "IntegerLiteral";
        case TokenType::FloatLiteral:       return "FloatLiteral";
        case TokenType::StringLiteral:      return "StringLiteral";
        case TokenType::Select:             return "SELECT";
        case TokenType::From:               return "FROM";
        case TokenType::Where:              return "WHERE";
        case TokenType::Group:              return "GROUP";
        case TokenType::By:                 return "BY";
        case TokenType::Having:             return "HAVING";
        case TokenType::Order:              return "ORDER";
        case TokenType::Asc:                return "ASC";
        case TokenType::Desc:               return "DESC";
        case TokenType::Limit:              return "LIMIT";
        case TokenType::Offset:             return "OFFSET";
        case TokenType::Join:               return "JOIN";
        case TokenType::Inner:              return "INNER";
        case TokenType::Left:               return "LEFT";
        case TokenType::On:                 return "ON";
        case TokenType::As:                 return "AS";
        case TokenType::And:                return "AND";
        case TokenType::Or:                 return "OR";
        case TokenType::Not:                return "NOT";
        case TokenType::Over:               return "OVER";
        case TokenType::Partition:          return "PARTITION";
        case TokenType::Rows:               return "ROWS";
        case TokenType::Between:            return "BETWEEN";
        case TokenType::Preceding:          return "PRECEDING";
        case TokenType::Current:            return "CURRENT";
        case TokenType::Row:                return "ROW";
        case TokenType::Like:               return "LIKE";
        case TokenType::In:                 return "IN";
        case TokenType::Is:                 return "IS";
        case TokenType::Null:               return "NULL";
        case TokenType::True:               return "TRUE";
        case TokenType::False:              return "FALSE";
        case TokenType::Plus:               return "+";
        case TokenType::Minus:              return "-";
        case TokenType::Star:               return "*";
        case TokenType::Slash:              return "/";
        case TokenType::Equal:              return "=";
        case TokenType::NotEqual:           return "!=";
        case TokenType::LessThan:           return "<";
        case TokenType::LessThanOrEqual:    return "<=";
        case TokenType::GreaterThan:        return ">";
        case TokenType::GreaterThanOrEqual: return ">=";
        case TokenType::Comma:              return ",";
        case TokenType::Semicolon:          return ";";
        case TokenType::Dot:                return ".";
        case TokenType::LeftParen:          return "(";
        case TokenType::RightParen:         return ")";
    }
    return "Unknown";
}

std::string Token::to_string() const {
    std::ostringstream oss;
    oss << "Token(" << token_type_name(type) << ", \"" << text << "\", " << line << ":" << column << ")";
    return oss.str();
}

} // namespace onyx::sql
