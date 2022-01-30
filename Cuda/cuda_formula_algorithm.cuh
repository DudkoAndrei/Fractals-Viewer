#pragma once

#include "../Algorithms/abstract_formula_algorithm.h"
#include "../Point/point.h"
#include "PolynomialCalculator/polynomial_calculator.cuh"

class CudaFormulaAlgorithm : public AbstractFormulaAlgorithm {
 public:
  void Calculate(
      std::vector<PointInfo>* point_info,
      const ImageSettings& settings,
      const std::vector<double>& expression) const override;
};
