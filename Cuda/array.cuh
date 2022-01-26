#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
class Array {
 public:
  __host__ __device__ Array(size_t size);

  __host__ __device__ const T& operator[](size_t idx) const;
  __host__ __device__ T& operator[](size_t idx);

  __host__ __device__ T* Data();
  __host__ __device__ const T* Data() const;

  __host__ __device__ ~Array();

 private:
  T* data_;
  size_t size_;
};

template<typename T>
__host__ __device__ Array<T>::Array(size_t size) : size_(size) {
  cudaMallocManaged(&data_, size * sizeof(T));
}

template<typename T>
__host__ __device__ Array<T>::~Array() {
  cudaFree(data_);
}

template<typename T>
__host__ __device__ const T& Array<T>::operator[](size_t idx) const {
  if (idx < size_) {
    return data_[idx];
  }
}

template<typename T>
__host__ __device__ T& Array<T>::operator[](size_t idx) {
  if (idx < size_) {
    return data_[idx];
  }
}

template<typename T>
__host__ __device__ T* Array<T>::Data() {
  return data_;
}

template<typename T>
__host__ __device__ const T* Array<T>::Data() const {
  return data_;
}
