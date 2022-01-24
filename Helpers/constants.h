#pragma once

#include <limits>

namespace constants {
  constexpr double kEps = 1e-4;
  constexpr long long kToIntMultiplier = 1 / kEps;
  constexpr double kMaxBoundAbsValue =
      std::numeric_limits<long long>::max() / kToIntMultiplier;
}
