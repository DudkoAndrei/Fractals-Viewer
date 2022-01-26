#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

template<typename T>
class Complex {
 public:
  CUDA_CALLABLE_MEMBER Complex() = default;
  CUDA_CALLABLE_MEMBER Complex(const T& real, const T& imag);
  CUDA_CALLABLE_MEMBER Complex(const T& real);

  CUDA_CALLABLE_MEMBER T& Real();
  CUDA_CALLABLE_MEMBER const T& Real() const;
  CUDA_CALLABLE_MEMBER T& Imag();
  CUDA_CALLABLE_MEMBER const T& Imag() const;

  CUDA_CALLABLE_MEMBER T Abs() const;

  CUDA_CALLABLE_MEMBER Complex<T> Conjugate() const;

  CUDA_CALLABLE_MEMBER Complex<T> Transpose() const;

  CUDA_CALLABLE_MEMBER Complex<T>& operator+=(const Complex& rhs);
  CUDA_CALLABLE_MEMBER Complex<T>& operator-=(const Complex& rhs);
  CUDA_CALLABLE_MEMBER Complex<T>& operator*=(const Complex& rhs);
  CUDA_CALLABLE_MEMBER Complex<T>& operator/=(const Complex& rhs);

  CUDA_CALLABLE_MEMBER Complex<T> operator+(const Complex& rhs) const;
  CUDA_CALLABLE_MEMBER Complex<T> operator-(const Complex& rhs) const;
  CUDA_CALLABLE_MEMBER Complex<T> operator*(const Complex& rhs) const;
  CUDA_CALLABLE_MEMBER Complex<T> operator/(const Complex& rhs) const;

  CUDA_CALLABLE_MEMBER Complex<T> operator-() const;

 private:
  T real_{0};
  T imaginary_{0};
};

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T>::Complex(const T& real, const T& imag)
    : real_(real), imaginary_(imag) {}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T>::Complex(const T& real) : Complex(real, 0) {}

template<typename T>
CUDA_CALLABLE_MEMBER T Complex<T>::Abs() const {
  return sqrt(real_ * real_ + imaginary_ * imaginary_);
}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T>& Complex<T>::operator+=(const Complex& rhs) {
  real_ += rhs.real_;
  imaginary_ += rhs.imaginary_;

  return *this;
}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T>& Complex<T>::operator-=(const Complex& rhs) {
  operator+=(-rhs);

  return *this;
}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T>& Complex<T>::operator*=(const Complex& rhs) {
  *this = Complex(real_ * rhs.real_ - imaginary_ * rhs.imaginary_,
                  real_ * rhs.imaginary_ + rhs.real_ * imaginary_);

  return *this;
}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T>& Complex<T>::operator/=(const Complex& rhs) {
  *this = Complex((real_ * rhs.real_ + imaginary_ * rhs.imaginary_)
                      / (rhs.real_ * rhs.real_
                          + rhs.imaginary_ * rhs.imaginary_),
                  (imaginary_ * rhs.real_ - real_ * rhs.imaginary_)
                      / (rhs.real_ * rhs.real_
                          + rhs.imaginary_ * rhs.imaginary_));

  return *this;
}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T> Complex<T>::operator-() const {
  return Complex(-real_, -imaginary_);
}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T> Complex<T>::operator+(
    const Complex& rhs) const {
  Complex result = *this;
  result += rhs;

  return result;
}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T> Complex<T>::operator-(
    const Complex& rhs) const {
  Complex result = *this;
  result -= rhs;

  return result;
}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T> Complex<T>::operator*(
    const Complex& rhs) const {
  Complex result = *this;
  result *= rhs;

  return result;
}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T> Complex<T>::operator/(
    const Complex& rhs) const {
  Complex result = *this;
  result /= rhs;

  return result;
}

template<typename T>
CUDA_CALLABLE_MEMBER T& Complex<T>::Real() {
  return real_;
}

template<typename T>
CUDA_CALLABLE_MEMBER const T& Complex<T>::Real() const {
  return real_;
}

template<typename T>
CUDA_CALLABLE_MEMBER T& Complex<T>::Imag() {
  return imaginary_;
}

template<typename T>
CUDA_CALLABLE_MEMBER const T& Complex<T>::Imag() const {
  return imaginary_;
}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T> Complex<T>::Conjugate() const {
  return Complex(real_, -imaginary_);
}

template<typename T>
CUDA_CALLABLE_MEMBER Complex<T> Complex<T>::Transpose() const {
  return Complex(imaginary_, real_);
}
