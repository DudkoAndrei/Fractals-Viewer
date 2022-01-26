#include "formulas_samples.cuh"

#include <cuComplex.h>

__global__ void GenerateMandelbrotBWPoint(
    bool* result,
    ImageSettings* settings) {

  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= settings->width * settings->height) {
    return;
  }

  uint64_t y = index / settings->width;
  uint64_t x = index % settings->width;

  cuDoubleComplex c = make_cuDoubleComplex(
      (x - settings->width / 2.0 - settings->offset_x) / settings->scale_x,
      (y - settings->height / 2.0 - settings->offset_y) / settings->scale_y);
  cuDoubleComplex z = make_cuDoubleComplex(0, 0);

  uint64_t iteration = 0;
  while (iteration < 1000 && cuCabs(z) < (2 << 8)) {
    z = cuCadd(cuCmul(z, z), c);

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
