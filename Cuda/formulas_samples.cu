#include "formulas_samples.cuh"
#include "complex.cuh"

__global__ void GenerateMandelbrotBWPoint(
    bool* result,
    ImageSettings* settings) {

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
    z = z * z + c;

    ++iteration;
  }

  if (iteration < 1000) {
    result[index] = true;
  } else {
    result[index] = false;
  }
}

void CudaMandelbrotBWSet(Array<bool>* data, const ImageSettings& settings) {
  uint64_t block_size = 256;
  uint64_t grid_size =
      (settings.width * settings.height + block_size - 1) / block_size;

  ImageSettings* d_settings;  // setting copy, stored in device memory
  cudaMalloc(&d_settings, sizeof(ImageSettings));
  cudaMemcpy(d_settings,
             &settings,
             sizeof(ImageSettings),
             cudaMemcpyHostToDevice);
  GenerateMandelbrotBWPoint<<<grid_size, block_size>>>(data->Data(),
                                                       d_settings);
  cudaDeviceSynchronize();

  cudaFree(d_settings);
}
