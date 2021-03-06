#include "formula_algorithm.h"

void FormulaAlgorithm::Calculate(
    std::vector<PointInfo>* point_info,
    const ImageSettings& settings,
    const Expression& expression) const {
  PolynomialCalculator<double> calc(expression);
  for (int y = 0; y < settings.height; ++y) {
    for (int x = 0; x < settings.width; ++x) {
      int index = y * settings.width + x;
      (*point_info)[index] = CalculatePoint(Point(x, y), settings, calc);
    }
  }
}

// TODO(niki4smirn): remove magic numbers
PointInfo FormulaAlgorithm::CalculatePoint(
    const Point& point,
    const ImageSettings& settings,
    const PolynomialCalculator<double>& calc) {
  Complex<double> c
      ((point.x() - settings.width / 2.0 - settings.offset_x) /
           settings.scale_x,
       (point.y() - settings.height / 2.0 - settings.offset_y) /
           settings.scale_y);
  Complex<double> z;

  constexpr size_t max_iterations = 1000;
  size_t iteration = 0;
  while (iteration < max_iterations && z.Abs() < (2 << 8)) {
    z = calc.Calculate(z) + c;
    ++iteration;
  }

  if (iteration < max_iterations) {
    return {iteration, Point(z.Real(), z.Imag())};
  }
  return {0, {}};
}
