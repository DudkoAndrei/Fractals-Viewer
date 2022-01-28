#include "formula_image_processor.h"

#include <utility>

#include "../Algorithms/formula_algorithm.h"
#include "../Cuda/cuda_formula_algorithm.cuh"

FormulaImageProcessor::FormulaImageProcessor() {
  gradient_ = Gradient(std::vector<Gradient::GradientPoint>{
      {0, Color{0, 0, 0}},
      {100000, Color{255, 255, 255}}});
}

FormulaImageProcessor::FormulaImageProcessor(Gradient gradient) :
    gradient_(std::move(gradient)) {}

Image FormulaImageProcessor::GenerateImage(
    bool use_cuda,
    const ImageSettings& settings) const {
  int width = settings.width;
  int height = settings.height;
  Image result(width, height);

  std::vector<PointInfo> points_info(width * height);

  std::unique_ptr<AbstractFormulaAlgorithm> algorithm;

  // dummy empty expression
  if (use_cuda) {
    algorithm = std::make_unique<CudaFormulaAlgorithm>();
  } else {
    algorithm = std::make_unique<FormulaAlgorithm>();
  }
  algorithm->Calculate(&points_info,
                       settings,
                       std::vector<Token>{{Token("1")}, {Token("0")},
                                          {Token("0")}});

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto[iter_val, point] = points_info[y * width + x];
      if (iter_val != 0) {
        result[y][x] = GetPointColorByIters(point, iter_val);
      } else {
        result[y][x] = Color{0, 0, 0};
      }
    }
  }
  return result;
}

Color FormulaImageProcessor::GetPointColorByIters(const Point& point,
                                                  uint64_t iters_count) const {
  double log_zn = log(point.x() * point.x() + point.y() * point.y()) / 2;
  double nu = log(log_zn / log(2)) / log(2);

  double iter = iters_count;
  iter += 1 - nu;

  iter *= 100;

  Color q = gradient_[iter];
  Color w = gradient_[iter + 1];

  double rate = iter - std::floor(iter);
  return q * rate + w * (1.0 - rate);
}
