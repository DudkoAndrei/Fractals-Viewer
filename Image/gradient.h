#pragma once

#include <vector>
#include <map>

#include "color.h"

class Gradient {
 public:
  struct GradientPoint {
    float coordinate{0};
    Color color;
  };

  Gradient() = default;
  explicit Gradient(const std::vector<Color>& colors);
  explicit Gradient(const std::vector<GradientPoint>& points);

  Color operator[](float coordinate) const;

  void AddPoint(const GradientPoint& point);

  float GetLeftBound() const;
  float GetRightBound() const;

 private:
  void FillPoints(const std::vector<GradientPoint>& points);

  static long long ToInnerCoordinate(float coordinate);
  static float FromInnerCoordinate(long long coordinate);

  std::map<long long, Color> points_;
};
