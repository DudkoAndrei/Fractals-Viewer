#pragma once

#include <memory>

#include "../ImageProcessor/formula_image_processor.h"

class Controller {
 public:
  void RunTest();
  void RunCudaTest();

 private:
  std::unique_ptr<FormulaImageProcessor> image_processor_;
};


