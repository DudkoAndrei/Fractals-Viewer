#include "array.cuh"
#include "cuda_formula_image_processor.cuh"
#include "formulas_samples.cuh"

Image CudaFormulaImageProcessor::GenerateImage(
    const ImageSettings& settings) const {
  Array<uint64_t> points_values(settings.width * settings.height);
  CudaBWFractal(&points_values, settings, Parse("conj(z) * conj(z) + c"));

  Image result(settings.width, settings.height);
  for (int y = 0; y < settings.height; ++y) {
    for (int x = 0; x < settings.width; ++x) {
      if (points_values[y * settings.width + x] != 0) {
        uint8_t temp = points_values[y * settings.width + x] * 16 - 1;
        result[y][x] = Color{temp, temp, temp};
      } else {
        result[y][x] = Color{0, 0, 0};
      }
    }
  }

  return result;
}
