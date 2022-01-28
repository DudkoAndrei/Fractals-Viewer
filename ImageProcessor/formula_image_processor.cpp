#include "formula_image_processor.h"

#include "../Algorithms/formula_algorithm.h"
#include "../Cuda/cuda_formula_algorithm.cuh"

FormulaImageProcessor::FormulaImageProcessor() {}

Image FormulaImageProcessor::GenerateImage(
    bool use_cuda,
    const ImageSettings& settings) const {
  int width = settings.width;
  int height = settings.height;
  Image result(width, height);
  std::vector<uint64_t> iters_count(width * height);
  std::unique_ptr<AbstractFormulaAlgorithm> algorithm;

  if (use_cuda) {
    algorithm = std::make_unique<CudaFormulaAlgorithm>();
  } else {
    algorithm = std::make_unique<FormulaAlgorithm>();
  }

  algorithm->Calculate(&iters_count,
                       settings,
                       std::vector<Token>{{Token("1")}, {Token("0")},
                                          {Token("0")}});

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (iters_count[y * settings.width + x] != 0) {
        uint8_t temp = iters_count[y * settings.width + x] * 16 - 1;
        result[y][x] = Color{temp, temp, temp};
      } else {
        result[y][x] = Color{0, 0, 0};
      }
    }
  }
  return result;
}
