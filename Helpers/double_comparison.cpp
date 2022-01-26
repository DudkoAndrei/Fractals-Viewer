#include "double_comparison.h"

#include <cmath>

bool helpers::double_comparison::IsEqual(double lhs, double rhs,
                                         double epsilon) {
  return std::fabs(lhs - rhs) <= epsilon;
}

bool helpers::double_comparison::IsNotEqual(double lhs, double rhs,
                                            double epsilon) {
  return !IsEqual(lhs, rhs, epsilon);
}

bool helpers::double_comparison::IsLess(double lhs, double rhs,
                                        double epsilon) {
  return (lhs + epsilon <= rhs);
}

bool helpers::double_comparison::IsLessOrEqual(double lhs, double rhs,
                                               double epsilon) {
  return IsLess(lhs, rhs, epsilon) || IsEqual(lhs, rhs, epsilon);
}

bool helpers::double_comparison::IsMore(double lhs, double rhs,
                                        double epsilon) {
  return (lhs - epsilon >= rhs);
}

bool helpers::double_comparison::IsMoreOrEqual(double lhs, double rhs,
                                               double epsilon) {
  return IsMore(lhs, rhs, epsilon) || IsEqual(lhs, rhs, epsilon);
}

bool helpers::double_comparison::IsInBounds(double l, double r, double x,
                                   double epsilon) {
  return IsLessOrEqual(l, x, epsilon) && IsLessOrEqual(x, r, epsilon);
}

double helpers::double_comparison::Min(double lhs, double rhs,
                                       double epsilon) {
  if (IsLess(lhs, rhs, epsilon)) {
    return lhs;
  }
  return rhs;
}

double helpers::double_comparison::Max(double lhs, double rhs,
                                       double epsilon) {
  return lhs + rhs - Min(lhs, rhs, epsilon);
}
