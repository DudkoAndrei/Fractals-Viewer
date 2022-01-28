#pragma once

#include <functional>
#include <optional>
#include <utility>

#include "abstract_formula_algorithm.h"
#include "../Point/point.h"
#include "../Cuda/PolynomialCalculator/polynomial_calculator.cuh"

class FormulaAlgorithm : public AbstractFormulaAlgorithm {
 public:
  void Calculate(
      std::vector<uint64_t>* iters_count,
      const ImageSettings& settings,
      const std::vector<Token>& expression) const override;

 private:
  uint64_t CalculatePoint(
      const Point& point,
      const PolynomialCalculator<double>& calc) const;
};
