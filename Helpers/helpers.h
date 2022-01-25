#pragma once

namespace helpers {
#include "constants.h"

template<typename T>
T MapValue(long double value,
           long double value_min,
           long double value_max,
           T map_min,
           T map_max) {
  value = std::max(std::min(value, value_max), value_min);
  if (value_max == value_min)
    return map_max;
  return (value - value_min) / (value_max - value_min) * (map_max - map_min) +
      map_min;
}
}  // namespace helpers
