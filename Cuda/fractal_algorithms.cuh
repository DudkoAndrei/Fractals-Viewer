#pragma once

#include "../Image/image_settings.h"
#include "../Point/point_info.h"

#include <vector>

void CudaBWFractal(
    std::vector<PointInfo>* data,
    const ImageSettings& settings,
    const std::vector<double>& expression);
