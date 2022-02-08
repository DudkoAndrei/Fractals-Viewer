#pragma once

#include <memory>

#include "../image.h"
#include "../image_settings.h"

class AbstractImageProcessor {
 public:
  virtual Image GenerateImage(
      bool use_cuda,
      const ImageSettings& settings) = 0;

  virtual ~AbstractImageProcessor() = default;
};


