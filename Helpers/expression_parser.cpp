#include "expression_parser.h"

#include <optional>
#include <sstream>
#include <stack>

std::optional<double> StringToDouble(const std::string& s) {
  std::istringstream input(s);
  float result;
  input >> result;
  if (input.eof() && !input.fail()) {
    return result;
  } else {
    return std::nullopt;
  }
}

Token::Token(const std::string& s) {
  if (s == "+") {
    type = TokenType::kPlus;
  } else if (s == "-") {
    type = TokenType::kMinus;
  } else if (s == "*") {
    type = TokenType::kMultiplication;
  } else if (s == "/") {
    type = TokenType::kDivision;
  } else if (s == "abs(" || s == "Abs(") {
    type = TokenType::kAbs;
  } else if (s == "conj(" || s == "Conj(") {
    type = TokenType::kConjugate;
  } else if (s == "trans(" || s == "Trans(") {
    type = TokenType::kTranspose;
  } else if (s == "(") {
    type = TokenType::kLeftBracket;
  } else if (s == ")") {
    type = TokenType::kRightBracket;
  } else if (s == "z" || s == "Z") {
    type = TokenType::kZ;
  } else if (s == "c" || s == "C") {
    type = TokenType::kC;
  } else {
    auto number = StringToDouble(s);

    if (number.has_value()) {
      type = TokenType::kConstant;
      value = number.value();
    } else {
      type = TokenType::kUnknown;
    }
  }
}

std::vector<Token> SplitToTokens(const std::string& s) {
  std::vector<Token> result;

  std::string temp;

  for (auto symbol : s) {
    switch (symbol) {
      case ' ': {
        if (!temp.empty()) {
          result.emplace_back(temp);
          temp.clear();
        }

        break;
      }
      case '(': {
        result.emplace_back(temp + symbol);
        temp.clear();

        break;
      }
      case ')': {
        if (!temp.empty()) {
          result.emplace_back(temp);
          temp.clear();
        }

        result.emplace_back(")");

        break;
      }
      default: {
        temp += symbol;
      }
    }
  }

  if (!temp.empty()) {
    result.emplace_back(temp);
  }

  return result;
}

int Priority(const Token& token) {
  switch (token.type) {
    case Token::TokenType::kLeftBracket: {
      return 6;
    }
    case Token::TokenType::kAbs:
    case Token::TokenType::kTranspose:
    case Token::TokenType::kConjugate: {
      return 5;
    }
    case Token::TokenType::kUnaryMinus: {
      return 4;
    }
    case Token::TokenType::kMultiplication:
    case Token::TokenType::kDivision: {
      return 3;
    }
    case Token::TokenType::kPlus:
    case Token::TokenType::kMinus: {
      return 2;
    }
    case Token::TokenType::kC:
    case Token::TokenType::kZ:
    case Token::TokenType::kConstant: {
      return 1;
    }
    case Token::TokenType::kRightBracket: {
      return 0;
    }
    default: {
      return -1;
    }
  }
}

std::vector<Token> InfixToPostfix(const std::vector<Token>& tokens) {
  std::vector<Token> result;
  std::stack<Token> operations;

  for (const Token& token : tokens) {
    switch (Priority(token)) {
      case 1: {
        result.push_back(token);
        break;
      }
      case 2:
      case 3:
      case 4:
      {
        if (operations.empty()
            || Priority(operations.top()) > 4
            || Priority(token) > Priority(operations.top())) {
          operations.push(token);
        } else if (Priority(token) <= Priority(operations.top())) {
          while (!operations.empty()
              && Priority(token) <= Priority(operations.top())
              && (Priority(operations.top()) < 5)) {
            result.push_back(operations.top());
            operations.pop();
          }

          operations.push(token);
        }
        break;
      }
      case 5:
      case 6: {
        operations.push(token);
        break;
      }
      case 0: {
        while (Priority(operations.top()) < 5) {
          result.push_back(operations.top());
          operations.pop();
        }

        if (Priority(operations.top()) == 5) {
          result.push_back(operations.top());
        }

        operations.pop();
        break;
      }
      default: {
        // Error
        break;
      }
    }
  }

  while (!operations.empty()) {
    result.push_back(operations.top());
    operations.pop();
  }

  return result;
}

std::vector<Token> Parse(const std::string& s) {
  return InfixToPostfix(SplitToTokens(s));
}
