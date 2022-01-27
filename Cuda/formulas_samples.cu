#include "complex.cuh"
#include "expression_calculator.cuh"
#include "formulas_samples.cuh"

__global__ void GenerateBWPoint(
    bool* result,
    ImageSettings* settings,
    Calculator<double>* calc) {
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
    result[index] = true;
  } else {
    result[index] = false;
  }
}

void CudaBWFractal(
    Array<bool>* data,
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

  Calculator<double> calc(d_expression, expression.size());
  Calculator<double>* d_calc;  // calculator copy, stored in device memory
  cudaMalloc(&d_calc, sizeof(Calculator<double>) );
  cudaMemcpy(d_calc,
             &calc,
             sizeof(Calculator<double>),
             cudaMemcpyHostToDevice);

  GenerateBWPoint<<<grid_size, block_size>>>(data->Data(),
                                             d_settings,
                                             d_calc);

  cudaDeviceSynchronize();

  cudaFree(d_settings);
}
