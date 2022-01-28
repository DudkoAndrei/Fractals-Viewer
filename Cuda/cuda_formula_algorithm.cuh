#pragma once

#include "../Algorithms/abstract_formula_algorithm.h"
#include "../Point/point.h"
#include "PolynomialCalculator/polynomial_calculator.cuh"

class CudaFormulaAlgorithm : public AbstractFormulaAlgorithm {
 public:
  void Calculate(
      std::vector<uint64_t>* iters_count,
      const ImageSettings& settings,
      const std::vector<Token>& expression) const override;
};
