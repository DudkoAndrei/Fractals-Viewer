#pragma once

#include <memory>

#include "../Algorithms/abstract_algorithm.h"
#include "../Image/image.h"
#include "../Image/image_settings.h"

class AbstractImageProcessor {
 public:
  virtual Image GenerateImage(const ImageSettings& settings) const = 0;

 private:
  std::unique_ptr<AbstractAlgorithm> algorithm_{nullptr};
};


