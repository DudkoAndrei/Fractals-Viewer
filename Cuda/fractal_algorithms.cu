#include "complex.cuh"
#include "fractal_algorithms.cuh"
#include "PolynomialCalculator/polynomial_calculator.cuh"

__global__ void GenerateBWPoint(
    uint64_t* result,
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
    z = calc->Calculate(z, c);
    ++iteration;
  }

  if (iteration < 1000) {
    result[index] = iteration % 16 + 1;
  } else {
    result[index] = 0;
  }
}

void CudaBWFractal(
    std::vector<uint64_t>* data,
    const ImageSettings& settings,
    const std::vector<Token>& expression) {
  uint64_t block_size = 256;
  uint64_t grid_size =
      (settings.width * settings.height + block_size - 1) / block_size;

  ImageSettings* d_settings;  // settings copy, stored in device memory
  cudaMalloc(&d_settings, sizeof(ImageSettings));
  cudaMemcpy(d_settings,
             &settings,
             sizeof(ImageSettings),
             cudaMemcpyHostToDevice);

  Token* d_expression;  // expression copy, stored in device memory
  cudaMalloc(&d_expression, sizeof(Token) * expression.size());
  cudaMemcpy(d_expression,
             expression.data(),
             sizeof(Token) * expression.size(),
             cudaMemcpyHostToDevice);

  PolynomialCalculator<double>* d_calc;
  PolynomialCalculator<double>  // calculator copy, stored in device memory
      calc(d_expression, expression.size());
  cudaMalloc(&d_calc, sizeof(PolynomialCalculator<double>));
  cudaMemcpy(d_calc,
             &calc,
             sizeof(PolynomialCalculator<double>),
             cudaMemcpyHostToDevice);

  uint64_t* d_data;  // array for data, stored in device memory
  cudaMalloc(&d_data, sizeof(uint64_t) * data->size());

  GenerateBWPoint<<<grid_size, block_size>>>(d_data,
                                             d_settings,
                                             d_calc);

  cudaDeviceSynchronize();

  cudaMemcpy(data->data(),
             d_data,
             sizeof(uint64_t) * data->size(),
             cudaMemcpyDeviceToHost);

  cudaFree(d_settings);
  cudaFree(d_expression);
  cudaFree(d_calc);
  cudaFree(d_data);
}
