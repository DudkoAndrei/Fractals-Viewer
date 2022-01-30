#include "cuda_formula_algorithm.cuh"
#include "fractal_algorithms.cuh"

void CudaFormulaAlgorithm::Calculate(
    std::vector<PointInfo>* point_info,
    const ImageSettings& settings,
    const std::vector<double>& expression) const {
  CudaBWFractal(point_info, settings, expression);
}
