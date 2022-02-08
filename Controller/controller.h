#pragma once

#include <memory>

#include "../Image/ImageProcessor/abstract_image_processor.h"

class Controller {
 public:
  void RunTest();
  void RunCudaTest();

 private:
  std::unique_ptr<AbstractImageProcessor> image_processor_;
};


