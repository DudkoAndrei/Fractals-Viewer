#include "array.cuh"
#include "cuda_formula_image_processor.cuh"
#include "formulas_samples.cuh"

Image CudaFormulaImageProcessor::GenerateImage(
    const ImageSettings& settings) const {
  Array<bool> points_values(settings.width * settings.height);
  CudaMandelbrotBWSet(&points_values, settings);

  Image result(settings.width, settings.height);
  for (int y = 0; y < settings.height; ++y) {
    for (int x = 0; x < settings.width; ++x) {
      result[y][x] = points_values[y * settings.width + x] ? Color{0, 0, 0} :
                     Color{255, 255, 255};
    }
  }

  return result;
}
