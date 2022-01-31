#include "cuda_error_handler.cuh"

#include <iostream>

void HandleCudaError(cudaError error) {
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << "\n";
  }
}
