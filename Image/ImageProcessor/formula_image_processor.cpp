#include "formula_image_processor.h"

#include <utility>

#include "../../Algorithms/formula_algorithm.h"
#include "../../Cuda/cuda_formula_algorithm.cuh"
#include "../../Helpers/MinMaxContainer/min_max_container.h"

FormulaImageProcessor::FormulaImageProcessor() {
  gradient_ = Gradient(std::vector<Color>{
      Color{0, 0, 0},
      Color{255, 255, 255}});
}

FormulaImageProcessor::FormulaImageProcessor(Gradient gradient) :
    gradient_(std::move(gradient)) {}

Image FormulaImageProcessor::GenerateImage(
    bool use_cuda,
    const ImageSettings& settings) {
  int width = settings.width;
  int height = settings.height;
  Image result(width, height);

  std::vector<PointInfo> points_info(width * height);

  std::unique_ptr<AbstractFormulaAlgorithm> algorithm;

  if (use_cuda) {
    algorithm = std::make_unique<CudaFormulaAlgorithm>();
  } else {
    algorithm = std::make_unique<FormulaAlgorithm>();
  }
  algorithm->Calculate(&points_info,
                       settings,
                       {{1, 0, 0}, {}, {}, {}});
  MinMaxContainer<double> min_max;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto[iter_val, point] = points_info[y * width + x];
      if (iter_val != 0) {
        min_max.Push(GetGradientPos(point, iter_val));
      }
    }
  }
  gradient_.Scale(log(min_max.GetMax() - min_max.GetMin()));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto[iter_val, point] = points_info[y * width + x];
      if (iter_val != 0) {
        result[y][x] = gradient_[log(GetGradientPos(point, iter_val))];
      } else {
        result[y][x] = {0, 0, 0};
      }
    }
  }
  return result;
}

// TODO(niki4smirn): remove magic numbers
double FormulaImageProcessor::GetGradientPos(const Point& point,
                                             uint64_t iters_count) {
  double log_zn = log(point.x() * point.x() + point.y() * point.y()) / 2;
  double nu = log(log_zn / log(2)) / log(2);

  double iter = iters_count;
  iter += 1 - nu;
  iter *= 2;

  return iter;
}
