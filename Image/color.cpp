#include "color.h"

#include <cassert>
#include <cmath>
#include <tuple>

#include "../Helpers/double_comparison.h"

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

uint8_t SubtractUnsigned(uint8_t lhs, uint8_t rhs) {
  if (lhs >= rhs) {
    return lhs - rhs;
  }
  return 0;
}

Color Color::operator-(const Color& rhs) const {
  Color result = *this;

  result.red_ = SubtractUnsigned(result.red_, rhs.red_);
  result.green_ = SubtractUnsigned(result.green_, rhs.green_);
  result.blue_ = SubtractUnsigned(result.blue_, rhs.blue_);

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

bool Color::operator==(const Color& rhs) const {
  return std::tie(red_, green_, blue_) ==
      std::tie(rhs.red_, rhs.green_, rhs.blue_);
}

bool Color::operator!=(const Color& rhs) const {
  return !((*this) == rhs);
}

Color Color::Mix(const Color& color1, const Color& color2, double alpha) {
  assert(helpers::double_comparison::IsInBounds(0, 1, std::fabs(alpha)));
  return color1 * alpha + (1 - alpha) * color2;
}
