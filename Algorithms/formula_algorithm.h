#pragma once

#include <functional>
#include <optional>
#include <utility>

#include "abstract_formula_algorithm.h"
#include "../Point/point.h"

class FormulaAlgorithm : AbstractFormulaAlgorithm {
 public:
  virtual void Calculate(
      std::vector<uint64_t>* iters_count,
      const ImageSettings& settings,
      const std::vector<Token>& expression) const;
};
