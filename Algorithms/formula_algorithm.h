#pragma once

#include <functional>
#include <optional>
#include <utility>

#include "../Point/point.h"

class FormulaAlgorithm {
 public:
  explicit FormulaAlgorithm(
      std::function<std::optional<long long>(const Point& point)> function);

  std::optional<long long> IterationsToConvergeCount(const Point& point) const;

 private:
  std::function<std::optional<long long>(const Point& point)> function_;
};
