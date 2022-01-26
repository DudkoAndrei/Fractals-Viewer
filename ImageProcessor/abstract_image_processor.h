#pragma once

#include <memory>

#include "../Image/image.h"
#include "../Image/image_settings.h"

class AbstractImageProcessor {
 public:
  virtual Image GenerateImage(const ImageSettings& settings) const = 0;
};


