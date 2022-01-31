#include "complex.cuh"
#include "fractal_algorithms.cuh"
#include "PolynomialCalculator/polynomial_calculator.cuh"
#include "cuda_error_handler.cuh"

// I don't know why, but if I move it to separate files, compilation fails
// (some Cuda tricks)

struct CudaPointInfo {
  __host__ __device__ CudaPointInfo() = default;
  __host__ __device__ CudaPointInfo(
      uint64_t iters_count,
      Complex<double> finish_point) :
      iters_count(iters_count), finish_point(finish_point) {}

  uint64_t iters_count{0};
  Complex<double> finish_point;
};

// TODO(niki4smirn): fix result type
__global__ void GenerateBWPoint(
    CudaPointInfo* result,
    ImageSettings* settings,
    PolynomialCalculator<double>* calc) {
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= settings->width * settings->height) {
    return;
  }

  uint64_t y = index / settings->width;
  uint64_t x = index % settings->width;

  Complex<double> c
      ((x - settings->width / 2.0 - settings->offset_x) / settings->scale_x,
       (y - settings->height / 2.0 - settings->offset_y) / settings->scale_y);
  Complex<double> z;

  uint64_t iteration = 0;
  while (iteration < 1000 && z.Abs() < (2 << 8)) {
    z = calc->Calculate(z) + c;
    ++iteration;
  }

  if (iteration < 1000) {
    result[index] = {iteration, z};
  } else {
    result[index] = {0, {}};
  }
}

void CudaBWFractal(
    std::vector<PointInfo>* data,
    const ImageSettings& settings,
    const std::vector<double>& expression) {
  uint64_t block_size = 256;
  uint64_t grid_size =
      (settings.width * settings.height + block_size - 1) / block_size;

  ImageSettings* d_settings;  // settings copy, stored in device memory
  HandleCudaError(cudaMalloc(&d_settings, sizeof(ImageSettings)));
  HandleCudaError(cudaMemcpy(d_settings,
                             &settings,
                             sizeof(ImageSettings),
                             cudaMemcpyHostToDevice));

  double* d_expression;  // expression copy, stored in device memory
  HandleCudaError(cudaMalloc(&d_expression,
                             sizeof(double) * expression.size()));
  HandleCudaError(cudaMemcpy(d_expression,
                             expression.data(),
                             sizeof(double) * expression.size(),
                             cudaMemcpyHostToDevice));

  PolynomialCalculator<double>* d_calc;
  PolynomialCalculator<double>  // calculator copy, stored in device memory
  calc(d_expression, expression.size());
  HandleCudaError(cudaMalloc(&d_calc, sizeof(PolynomialCalculator<double>)));
  HandleCudaError(cudaMemcpy(d_calc,
                             &calc,
                             sizeof(PolynomialCalculator<double>),
                             cudaMemcpyHostToDevice));

  CudaPointInfo* d_data;  // array for data, stored in device memory
  HandleCudaError(cudaMalloc(&d_data, sizeof(CudaPointInfo) * data->size()));

  GenerateBWPoint<<<grid_size, block_size>>>(d_data,
                                             d_settings,
                                             d_calc);

  HandleCudaError(cudaDeviceSynchronize());

  std::vector<CudaPointInfo> temp_data(settings.width * settings.height);

  HandleCudaError(cudaMemcpy(temp_data.data(),
                             d_data,
                             sizeof(CudaPointInfo) * data->size(),
                             cudaMemcpyDeviceToHost));

  int pos = 0;
  for (auto[iters, point] : temp_data) {
    (*data)[pos] = {iters, {point.Real(), point.Imag()}};
    ++pos;
  }

  HandleCudaError(cudaFree(d_settings));
  HandleCudaError(cudaFree(d_expression));
  HandleCudaError(cudaFree(d_calc));
  HandleCudaError(cudaFree(d_data));
}
