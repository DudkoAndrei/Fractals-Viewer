#include "cuda_formula_algorithm.cuh"
#include "fractal_algorithms.cuh"

void CudaFormulaAlgorithm::Calculate(
    std::vector<uint64_t>* iters_count,
    const ImageSettings& settings,
    const std::vector<Token>& expression) const {
  CudaBWFractal(iters_count, settings, expression);
}
