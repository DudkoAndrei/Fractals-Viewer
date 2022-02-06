#pragma once

#include <vector>
#include <map>

#include "color.h"

class Gradient {
 public:
  struct GradientPoint {
    double coordinate{0};
    Color color;
  };

  Gradient() = default;
  explicit Gradient(const std::vector<Color>& colors);
  explicit Gradient(const std::vector<GradientPoint>& points);

  Color operator[](double coordinate) const;

  void AddPoint(double coordinate, const Color& color);
  void AddPoint(const GradientPoint& point);

  void Scale(double new_len);

  double GetLeftBound() const;
  double GetRightBound() const;

 private:
  void AddPoints(const std::vector<GradientPoint>& points);

  std::map<double, Color> points_;
};
