#pragma once

#include <cstdint>
#include <vector>

#include "../Cuda/PolynomialCalculator/expression.h"
#include "../Image/image_settings.h"
#include "../Point/point_info.h"

class AbstractFormulaAlgorithm {
 public:
  virtual void Calculate(
      std::vector<PointInfo>* iters_count,
      const ImageSettings& settings,
      const Expression& expression) const = 0;

  virtual ~AbstractFormulaAlgorithm() = default;
};


