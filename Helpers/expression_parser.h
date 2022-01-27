#pragma once

#include <string>
#include <vector>

struct Token {
  enum class TokenType {
    kUnknown,
    kPlus,
    kMinus,
    kMultiplication,
    kDivision,
    kUnaryMinus,
    kAbs,
    kTranspose,
    kConjugate,
    kLeftBracket,
    kRightBracket,
    kZ,
    kC,
    kConstant
  };
  explicit Token(const std::string& s);

  TokenType type{TokenType::kUnknown};
  double value{0};
};

std::vector<Token> Parse(const std::string& s);
