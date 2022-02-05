#include "cuda_formula_algorithm.cuh"
#include "fractal_algorithms.cuh"

void CudaFormulaAlgorithm::Calculate(
    std::vector<PointInfo>* point_info,
    const ImageSettings& settings,
    const Expression& expression) const {
  // why we call here just some function from somewhere?
  CudaBWFractal(point_info, settings, expression);
}
