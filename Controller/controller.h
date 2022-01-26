#pragma once

#include <memory>

#include "../Cuda/cuda_formula_image_processor.cuh"
#include "../ImageProcessor/formula_image_processor.h"

class Controller {
 public:
  void RunTest();
  void RunCudaTest();

 private:
  std::unique_ptr<FormulaImageProcessor> image_processor_;
  std::unique_ptr<CudaFormulaImageProcessor> cuda_image_processor_;
};


