#include "gradient.h"

#include <utility>
#include <cassert>
#include <cmath>

#include "../Helpers/helpers.h"

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
    points_[ToInnerCoordinate(coordinate)] = color;
  }
}

Color Gradient::operator[](float coordinate) const {
  using namespace helpers;
  assert(std::fabs(coordinate) <= constants::kMaxBoundAbsValue);
  long long new_coordinate = ToInnerCoordinate(coordinate);
  auto closest_right = points_.upper_bound(new_coordinate);
  auto closest_left = points_.lower_bound(new_coordinate);
  --closest_left;
  assert(closest_left != points_.end());
  assert(closest_right != points_.end());

  return MapValue(new_coordinate, closest_left->first, closest_right->first,
                  closest_left->second, closest_right->second);
}

float Gradient::GetLeftBound() const {
  assert(!points_.empty());
  return FromInnerCoordinate(points_.begin()->first);
}

float Gradient::GetRightBound() const {
  assert(!points_.empty());
  return FromInnerCoordinate((--points_.end())->first);
}

long long Gradient::ToInnerCoordinate(float coordinate) {
  return coordinate * helpers::constants::kToIntMultiplier;
}

float Gradient::FromInnerCoordinate(long long coordinate) {
  return static_cast<float>(coordinate) / helpers::constants::kToIntMultiplier;
}

void Gradient::AddPoint(const Gradient::GradientPoint& point) {
  points_[ToInnerCoordinate(point.coordinate)] = point.color;
}
