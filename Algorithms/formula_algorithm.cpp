#include "formula_algorithm.h"

#include <complex>

void FormulaAlgorithm::Calculate(
    std::vector<uint64_t>* iters_count,
    const ImageSettings& settings,
    const std::vector<Token>& expression) const {
  PolynomialCalculator<double> calc(expression);
  for (int y = 0; y < settings.height; ++y) {
    for (int x = 0; x < settings.width; ++x) {
      (*iters_count)[y * settings.width + x] = CalculatePoint(Point(x, y),
                                                              calc);
    }
  }
}

// TODO(niki4smirn): remove magic numbers
uint64_t FormulaAlgorithm::CalculatePoint(
    const Point& point,
    const PolynomialCalculator<double>& calc) const {
  std::complex<double> c((point.x() - 500) / 200.0,
                         (point.y() - 500) / 200.0);
  std::complex<double> z(0.0, 0.0);

  size_t iteration = 0;
  while (iteration < 1000 && abs(z) < (2 << 8)) {
    z = calc.Calculate(z, c);
    ++iteration;
  }

  if (iteration < 1000) {
    return iteration;
  }
  return 0;
}
