#include "point.h"

#include "../Helpers/helpers.h"

using helpers::double_comparison::IsEqual;

Point::Point(double x, double y) : x_(x), y_(y) {}

double Point::x() const {
  return x_;
}

double& Point::x() {
  return x_;
}

double Point::y() const {
  return y_;
}

double& Point::y() {
  return y_;
}

Point Point::operator+(const Point& other) const {
  return {x_ + other.x_, y_ + other.y_};
}

Point Point::operator-(const Point& other) const {
  return {x_ - other.x_, y_ - other.y_};
}

bool Point::operator==(const Point& other) const {
  return IsEqual(x_, other.x_) && IsEqual(y_, other.y_);
}

bool Point::operator!=(const Point& other) const {
  return !((*this) == other);
}
