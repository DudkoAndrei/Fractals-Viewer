#pragma once

#include "abstract_image_processor.h"
#include "../Cuda/cuda_formula_algorithm.cuh"
#include "../Algorithms/formula_algorithm.h"

#include <complex>

class FormulaImageProcessor : AbstractImageProcessor {
 public:
  template<typename... Args>
  explicit FormulaImageProcessor(Args... args);
  virtual Image GenerateImage(
      bool use_cuda,
      const ImageSettings& settings) const;

 private:
  std::unique_ptr<FormulaAlgorithm> algorithm_;
  std::unique_ptr<CudaFormulaAlgorithm> cuda_algorithm_;
};

template<typename... Args>
FormulaImageProcessor::FormulaImageProcessor(Args... args) {
    algorithm_ = std::make_unique<FormulaAlgorithm>(args...);
    cuda_algorithm_ = std::make_unique<CudaFormulaAlgorithm>(args...);
}
