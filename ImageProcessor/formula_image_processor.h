#pragma once

#include "abstract_image_processor.h"

#include <complex>

#include "../Image/gradient.h"
#include "../Point/point.h"

class FormulaImageProcessor : public AbstractImageProcessor {
 public:
  // here will be some parameters soon
  FormulaImageProcessor();
  explicit FormulaImageProcessor(Gradient gradient);
  Image GenerateImage(
      bool use_cuda,
      const ImageSettings& settings) override;

 private:
  static double GetGradientPos(const Point& point, uint64_t iters_count);

  Gradient gradient_;
};
