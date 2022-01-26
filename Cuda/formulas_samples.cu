#include "formulas_samples.cuh"

#include <cuComplex.h>

__global__ void GenerateMandelbrotBWPoint(
    bool* result,
    const uint64_t width,
    const uint64_t height,
    const uint64_t offset_x,
    const uint64_t offset_y,
    const double scale_x,
    const double scale_y) {

  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= width * height) {
    return;
  }

  uint64_t y = index / width;
  uint64_t x = index % width;

  cuDoubleComplex c = make_cuDoubleComplex(
      (x - width / 2.0 - offset_x) / scale_x,
      (y - height / 2.0 - offset_y) / scale_y);
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

void CudaMandelbrotBWSet(Array<bool>* data, ImageSettings settings) {
  uint64_t block_size = 256;
  uint64_t grid_size =
      (settings.width * settings.height + block_size - 1) / block_size;
  GenerateMandelbrotBWPoint<<<grid_size, block_size>>>(data->Data(),
                                      settings.width,
                                      settings.height,
                                      settings.offset_x,
                                      settings.offset_y,
                                      settings.scale_x,
                                      settings.scale_y);
  cudaDeviceSynchronize();
}
