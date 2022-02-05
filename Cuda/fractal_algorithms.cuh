#pragma once

#include <vector>

#include "../Image/image_settings.h"
#include "../Point/point_info.h"
#include "PolynomialCalculator/expression.h"

void CudaBWFractal(
    std::vector<PointInfo>* data,
    const ImageSettings& settings,
    const Expression& expression);
