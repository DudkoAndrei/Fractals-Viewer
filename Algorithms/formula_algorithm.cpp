#include "formula_algorithm.h"

FormulaAlgorithm::FormulaAlgorithm(
    std::function<std::optional<long long>(const Point& point)> function) :
    function_(std::move(function)) {}

std::optional<long long> FormulaAlgorithm::IterationsToConvergeCount(
    const Point& point) const {
  return function_(point);
}
