#include "gradient.h"

#include <cassert>

#include "../Helpers/double_comparison.h"

Gradient::Gradient(const std::vector<Color>& colors) {
  std::vector<GradientPoint> points;
  float pos = 0;
  for (const auto& color : colors) {
    points.push_back({pos, color});
    ++pos;
  }
  FillPoints(points);
}

Gradient::Gradient(const std::vector<GradientPoint>& points)  {
  FillPoints(points);
}

void Gradient::FillPoints(const std::vector<GradientPoint>& points) {
  for (const auto& [coordinate, color] : points) {
    points_[coordinate] = color;
  }
}

Color Gradient::operator[](double coordinate) const {
  auto closest_left = points_.lower_bound(coordinate);
  assert(closest_left != points_.end());
  auto closest_right = closest_left;
  if (helpers::double_comparison::IsLess(coordinate, closest_left->first)) {
    --closest_left;
  } else {
    ++closest_right;
  }

  double alpha = 0;
  if (closest_left->first != closest_right->first) {
    alpha = (closest_right->first - coordinate) /
        (closest_right->first - closest_left->first);
  }
  assert(closest_right != points_.end() ||
      helpers::double_comparison::IsEqual(alpha, 1));
  return Color::Mix(closest_left->second, closest_right->second, alpha);
}

double Gradient::GetLeftBound() const {
  assert(!points_.empty());
  return points_.begin()->first;
}

double Gradient::GetRightBound() const {
  assert(!points_.empty());
  return (--points_.end())->first;
}

void Gradient::AddPoint(const Gradient::GradientPoint& point) {
  points_[point.coordinate] = point.color;
}
