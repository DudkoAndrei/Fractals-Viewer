#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#include <cmath>
using std::fabs;
#endif

#include "../complex.cuh"
#include "expression.h"

template<typename T>
class PolynomialCalculator {
 public:
  PolynomialCalculator() = default;
  CUDA_CALLABLE_MEMBER explicit PolynomialCalculator(
      const Expression& polynomial);

  CUDA_CALLABLE_MEMBER PolynomialCalculator(
      const T* expression,
      const expression::AllSegments& segments);

  CUDA_CALLABLE_MEMBER Complex<T> Calculate(Complex<T> z) const;

  CUDA_CALLABLE_MEMBER const T* Data() const;
  CUDA_CALLABLE_MEMBER size_t Size() const;

 private:
  CUDA_CALLABLE_MEMBER Complex<T> Calculate(
      Complex<T> z, expression::Segment segment) const;

  const T* expression_{nullptr};

  expression::AllSegments segments_;
};

template<typename T>
CUDA_CALLABLE_MEMBER PolynomialCalculator<T>::PolynomialCalculator(
    const Expression& polynomial) :
    expression_(polynomial.GetData()),
    segments_(polynomial.GetSegments()) {}

template<typename T>
CUDA_CALLABLE_MEMBER PolynomialCalculator<T>::PolynomialCalculator(
    const T* expression,
    const expression::AllSegments& segments) :
    expression_(expression), segments_(segments) {}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T> PolynomialCalculator<T>::Calculate(
    Complex<T> z) const {
  Complex<T> result;
  if (segments_.default_segment.len != 0) {
    result += Calculate(z, segments_.default_segment);
  }
  if (segments_.conjugate_segment.len != 0) {
    result += Calculate(z.Conjugate(), segments_.conjugate_segment);
  }
  if (segments_.transpose_segment.len != 0) {
    result += Calculate(z.Transpose(), segments_.transpose_segment);
  }
  if (segments_.absolute_segment.len != 0) {
    result += Calculate(Complex<T>(fabs(z.Real()), fabs(z.Imag())),
        segments_.absolute_segment);
  }
  return result;
}

template<typename T>
CUDA_CALLABLE_MEMBER const T* PolynomialCalculator<T>::Data() const {
  return expression_;
}

template<typename T>
CUDA_CALLABLE_MEMBER size_t PolynomialCalculator<T>::Size() const {
  return segments_.default_segment.len +
      segments_.conjugate_segment.len +
      segments_.transpose_segment.len +
      segments_.absolute_segment.len;
}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T> PolynomialCalculator<T>::Calculate(
    Complex<T> z,
    expression::Segment segment) const {
  Complex<T> result;
  for (int i = segment.start; i < segment.start + segment.len; ++i) {
    result *= z;
    result += expression_[i];
  }
  return result;
}
