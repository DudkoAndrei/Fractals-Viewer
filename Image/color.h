#pragma once

#include <cstdint>

class Color {
 public:
  Color() = default;
  Color(uint8_t red, uint8_t green, uint8_t blue);

  uint8_t r() const;
  uint8_t g() const;
  uint8_t b() const;

  uint8_t& r();
  uint8_t& g();
  uint8_t& b();

  Color operator+(const Color& rhs) const;
  Color operator-(const Color& rhs) const;
  Color operator*(double rhs) const;
  Color operator/(double rhs) const;

  friend Color operator*(double lhs, const Color& rhs);

  bool operator==(const Color& rhs) const;

 private:
  uint8_t red_{0};
  uint8_t green_{0};
  uint8_t blue_{0};
};

