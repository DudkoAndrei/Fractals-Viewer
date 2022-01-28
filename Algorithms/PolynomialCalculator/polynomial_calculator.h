#pragma once

#include <complex>
#include <utility>
#include <vector>

#include "../../Helpers/expression_parser.h"

template<typename T>
class PolynomialCalculator {
 public:
  PolynomialCalculator() = default;
  explicit PolynomialCalculator(std::vector<Token> polynomial_expression);

  std::complex<T> Calculate(
      const std::complex<T>& z,
      const std::complex<T>& c) const;

 private:
  std::vector<Token> expression_;
};

template<typename T>
PolynomialCalculator<T>::PolynomialCalculator(
    std::vector<Token> polynomial_expression) :
    expression_(std::move(polynomial_expression)) {}

// dummy implementation
template<typename T>
std::complex<T> PolynomialCalculator<T>::Calculate(
    const std::complex<T>& z,
    const std::complex<T>& c) const {
  return z * z + c;
}


