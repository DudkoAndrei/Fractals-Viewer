#pragma once

#include "abstract_image_processor.h"

#include <complex>

#include "../Algorithms/formula_algorithm.h"

class FormulaImageProcessor : AbstractImageProcessor {
 public:
  template<typename... Args>
  explicit FormulaImageProcessor(Args... args);
  virtual Image GenerateImage(const ImageSettings& settings) const;

 private:
  std::unique_ptr<FormulaAlgorithm> algorithm_;
};

template<typename... Args>
FormulaImageProcessor::FormulaImageProcessor(Args... args) {
  algorithm_ = std::make_unique<FormulaAlgorithm>(args...);
}
