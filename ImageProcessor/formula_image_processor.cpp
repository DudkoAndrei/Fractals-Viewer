#include "formula_image_processor.h"

Image FormulaImageProcessor::GenerateImage(
    const ImageSettings& settings) const {
  int width = settings.width;
  int height = settings.height;
  Image result(width, height);

  std::vector<uint64_t> iters_count(width * height);
  // dummy empty expression
  algorithm_->Calculate(&iters_count, settings, std::vector<Token>{});

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint64_t iter_val = iters_count[y * width + x];
      result[y][x] = (iter_val != 0) ? Color{0, 0, 0} : Color{255, 255, 255};
    }
  }
  return result;
}
