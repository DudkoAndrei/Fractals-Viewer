#pragma once

#include <cstdint>

#include "point.h"

struct PointInfo {
  uint64_t iters_count{0};
  Point finish_point;
};
