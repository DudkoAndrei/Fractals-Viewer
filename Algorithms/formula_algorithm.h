#pragma once

#include <functional>
#include <optional>
#include <utility>

#include "abstract_formula_algorithm.h"
#include "../Point/point_info.h"
#include "../Cuda/PolynomialCalculator/polynomial_calculator.cuh"

class FormulaAlgorithm final : public AbstractFormulaAlgorithm {
 public:
  void Calculate(
      std::vector<PointInfo>* iters_count,
      const ImageSettings& settings,
      const Expression& expression) const override;

 private:
  static PointInfo CalculatePoint(
      const Point& point,
      const ImageSettings& settings,
      const PolynomialCalculator<double>& calc);
};
