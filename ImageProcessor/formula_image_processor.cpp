#include "formula_image_processor.h"

Image FormulaImageProcessor::GenerateImage(
    const ImageSettings& settings) const {
  int width = settings.width;
  int height = settings.height;
  Image result(width, height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // auto opt = algorithm_->IterationsToConvergeCount(Point(x, y));
      // result[y][x] = opt.has_value() ? Color{0, 0, 0} :
      //                Color{255, 255, 255};
    }
  }
  return result;
}
