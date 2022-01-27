#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include "complex.cuh"
#include "../Helpers/expression_parser.h"
#include "stack.cuh"

template<typename T>
class Calculator {
 public:
  CUDA_CALLABLE_MEMBER Calculator(Token* expression, size_t length);
  CUDA_CALLABLE_MEMBER Complex<T> Calculate(
      const Complex<T>& z,
      const Complex<T>& c);
 private:
  Token* expression_;
  size_t expression_length_;
};

template<typename T>
CUDA_CALLABLE_MEMBER Calculator<T>::Calculator(Token* expression, size_t length)
    : expression_(expression), expression_length_(length) {}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T> Calculator<T>::Calculate(
    const Complex<T>& z,
    const Complex<T>& c) {
  Stack<Complex<T>> result;

  for (size_t i = 0; i < expression_length_; ++i) {
    switch (expression_[i].type) {
      case Token::TokenType::kZ: {
        result.Push(z);

        break;
      }
      case Token::TokenType::kC: {
        result.Push(c);

        break;
      }
      case Token::TokenType::kConstant: {
        result.Push(Complex<double>(expression_[i].value));

        break;
      }
      case Token::TokenType::kAbs: {
        result.Top() = Complex<double>(result.Top());

        break;
      }
      case Token::TokenType::kTranspose: {
        result.Top() = result.Top().Transpose();

        break;
      }
      case Token::TokenType::kConjugate: {
        result.Top() = result.Top().Conjugate();

        break;
      }
      case Token::TokenType::kUnaryMinus: {
        result.Top() = -result.Top();

        break;
      }
      case Token::TokenType::kPlus: {
        Complex<double> temp = result.Top();
        result.Pop();
        result.Top() += temp;

        break;
      }
      case Token::TokenType::kMinus: {
        Complex<double> temp = result.Top();
        result.Pop();
        result.Top() -= temp;

        break;
      }
      case Token::TokenType::kMultiplication: {
        Complex<double> temp = result.Top();
        result.Pop();
        result.Top() *= temp;

        break;
      }
      case Token::TokenType::kDivision: {
        Complex<double> temp = result.Top();
        result.Pop();
        result.Top() /= temp;

        break;
      }
      default: {
        // Error

        break;
      }
    }
  }

  return result.Top();
}
