#include "color.h"

#include <tuple>

Color::Color(uint8_t red, uint8_t green, uint8_t blue)
    : red_(red), green_(green), blue_(blue) {}

uint8_t Color::r() const {
  return red_;
}

uint8_t Color::g() const {
  return green_;
}

uint8_t Color::b() const {
  return blue_;
}

uint8_t& Color::r() {
  return red_;
}

uint8_t& Color::g() {
  return green_;
}

uint8_t& Color::b() {
  return blue_;
}

Color Color::operator+(const Color& rhs) const {
  Color result = *this;

  result.red_ += rhs.red_;
  result.green_ += rhs.green_;
  result.blue_ += rhs.blue_;

  return result;
}

Color Color::operator-(const Color& rhs) const {
  Color result = *this;

  result.red_ -= rhs.red_;
  result.green_ -= rhs.green_;
  result.blue_ -= rhs.blue_;

  return result;
}

Color Color::operator*(double rhs) const {
  Color result = *this;

  result.red_ *= rhs;
  result.green_ *= rhs;
  result.blue_ *= rhs;

  return result;
}

Color operator*(double lhs, const Color& rhs) {
  return rhs * lhs;
}

Color Color::operator/(double rhs) const {
  return (*this) * (1 / rhs);
}
