#pragma once

namespace helpers {
#include "constants.h"

template<typename T1, typename T2>
T2 MapValue(T1 value, T1 value_min, T1 value_max, T2 map_min, T2 map_max) {
  value = std::max(std::min(value, value_max), value_min);
  if (value_max == value_min)
    return map_max;
  return (value - value_min) * (map_max - map_min) / (value_max - value_min) +
      map_min;
}
}  // namespace helpers