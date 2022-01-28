#pragma once

#include <cstdint>
#include <vector>

#include "../Image/image_settings.h"
#include "../Helpers/expression_parser.h"

class AbstractFormulaAlgorithm {
 public:
  virtual void Calculate(
      std::vector<uint64_t>* iters_count,
      const ImageSettings& settings,
      const std::vector<Token>& expression) const = 0;
};


