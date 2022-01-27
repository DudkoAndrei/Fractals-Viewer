#pragma once

#include "../Helpers/expression_parser.h"
#include "../Image/image_settings.h"
#include "array.cuh"

void CudaBWFractal(
    Array<bool>* data,
    const ImageSettings& settings,
    const std::vector<Token>& expression);
