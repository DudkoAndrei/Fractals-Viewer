#include "formula_image_processor.h"

Image FormulaImageProcessor::GenerateImage(
    const ImageSettings& settings) const {
  int width = settings.width;
  int height = settings.height;
  Image result(width, height);

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      auto opt = algorithm_->IterationsToConvergeCount(
          Point(i, j));
      result.Get(i, j) = opt.has_value() ? Color{0, 0, 0} :
          Color{255, 255, 255};
    }
  }
  return result;
}
