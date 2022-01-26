#pragma once

#include "../Image/image_settings.h"
#include "array.cuh"

void CudaMandelbrotBWSet(Array<bool>* data, const ImageSettings& settings);
