#pragma once

#include <cstdint>
#include <vector>

#include "../Image/image_settings.h"
#include "../Helpers/expression_parser.h"
#include "../Point/point_info.h"

class AbstractFormulaAlgorithm {
 public:
  virtual void Calculate(
      std::vector<PointInfo>* iters_count,
      const ImageSettings& settings,
      const std::vector<double>& expression) const = 0;
};


