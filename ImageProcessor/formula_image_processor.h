#pragma once

#include "abstract_image_processor.h"

#include <complex>

#include "../Algorithms/abstract_formula_algorithm.h"

class FormulaImageProcessor : public AbstractImageProcessor {
 public:
  // here will be some parameters soon
  FormulaImageProcessor();
  Image GenerateImage(
      bool use_cuda,
      const ImageSettings& settings) const override;
};
