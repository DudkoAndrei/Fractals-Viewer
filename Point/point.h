#pragma once

class Point {
 public:
  Point() = default;
  Point(double x, double y);

  double x() const;
  double& x();

  double y() const;
  double& y();

  Point operator+(const Point& other) const;
  Point operator-(const Point& other) const;

  bool operator==(const Point& other) const;
  bool operator!=(const Point& other) const;

 private:
  double x_{0};
  double y_{0};
};
