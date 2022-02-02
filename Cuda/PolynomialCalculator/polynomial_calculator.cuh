#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#include <cmath>
using std::fabs;
#endif

#include "../complex.cuh"

namespace polynomial {

enum class Type {
  kDefault,
  kConjugate,
  kTranspose,
  kAbsolute  // z = |z.Real()| + i|z.Imag()|
};

}  // namespace polynomial

template<typename T>
class PolynomialCalculator {
 public:
  explicit PolynomialCalculator(
      const std::vector<double>& polynomial,
      polynomial::Type type = polynomial::Type::kDefault);

  CUDA_CALLABLE_MEMBER PolynomialCalculator(
      const T* expression,
      size_t size,
      polynomial::Type type = polynomial::Type::kDefault);

  CUDA_CALLABLE_MEMBER Complex<T> Calculate(
      Complex<T> z) const;

  CUDA_CALLABLE_MEMBER const T* Data() const;
  CUDA_CALLABLE_MEMBER size_t Size() const;

 private:
  const T* expression_;
  size_t size_;
  polynomial::Type type_;
};

template<typename T>
CUDA_CALLABLE_MEMBER PolynomialCalculator<T>::PolynomialCalculator(
    const T* expression,
    size_t size,
    polynomial::Type type)
    : expression_(expression), size_(size), type_(type) {}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T> PolynomialCalculator<T>::Calculate(
    Complex<T> z) const {
  Complex<T> result;
  switch (type_) {
    case polynomial::Type::kConjugate: {
      z = z.Conjugate();

      break;
    }
    case polynomial::Type::kTranspose: {
      z = z.Transpose();

      break;
    }
    case polynomial::Type::kAbsolute: {
      z = Complex<T>(fabs(z.Real()), fabs(z.Imag()));

      break;
    }
    default: {
      break;
    }
  }

  for (size_t i = 0; i < size_; ++i) {
    result *= z;
    result += expression_[i];
  }

  return result;
}

template<typename T>
CUDA_CALLABLE_MEMBER const T* PolynomialCalculator<T>::Data() const {
  return expression_;
}

template<typename T>
CUDA_CALLABLE_MEMBER size_t PolynomialCalculator<T>::Size() const {
  return size_;
}

template<typename T>
PolynomialCalculator<T>::PolynomialCalculator(
    const std::vector<double>& polynomial,
    polynomial::Type type)
    : expression_(polynomial.data()), size_(polynomial.size()), type_(type) {}

