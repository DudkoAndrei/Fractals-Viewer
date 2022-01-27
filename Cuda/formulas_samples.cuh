#pragma once

#include "../Helpers/expression_parser.h"
#include "../Image/image_settings.h"
#include "array.cuh"

void CudaBWFractal(
    Array<uint64_t>* data,
    const ImageSettings& settings,
    const std::vector<Token>& expression);
