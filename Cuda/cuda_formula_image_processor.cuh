#pragma once

#include "../ImageProcessor/abstract_image_processor.h"

class CudaFormulaImageProcessor : public AbstractImageProcessor {
 public:
  CudaFormulaImageProcessor() = default;
  virtual Image GenerateImage(const ImageSettings& settings) const;
};
