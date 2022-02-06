#pragma once

#include "constants.h"

namespace helpers::double_comparison {
  bool IsEqual(double lhs, double rhs, double epsilon = constants::kEps);
  bool IsNotEqual(double lhs, double rhs, double epsilon = constants::kEps);
  bool IsLess(double lhs, double rhs, double epsilon = constants::kEps);
  bool IsMore(double lhs, double rhs, double epsilon = constants::kEps);

  bool IsLessOrEqual(double lhs, double rhs, double epsilon = constants::kEps);
  bool IsMoreOrEqual(double lhs, double rhs, double epsilon = constants::kEps);

  bool IsInBounds(
      double l, double r, double x, double epsilon = constants::kEps);

  double Min(double lhs, double rhs, double epsilon = constants::kEps);
  double Max(double lhs, double rhs, double epsilon = constants::kEps);
}  // namespace helpers::double_comparison
