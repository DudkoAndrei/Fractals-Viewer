#pragma once

#include "abstract_algorithm.h"

#include <functional>
#include <optional>

template<typename T>
class FormulaAlgorithm : public AbstractAlgorithm {
 public:
  explicit FormulaAlgorithm(
      std::function<std::optional<long long>(T)> function);

  std::optional<long long> IterationsToConvergeCount(T point) const;

  AlgorithmType GetType() const override;

 private:
  std::function<std::optional<long long>(T)> function_;
};

template<typename T>
FormulaAlgorithm<T>::FormulaAlgorithm(
    std::function<std::optional<long long>(T)> function) :
    function_(function) {}

template<typename T>
std::optional<long long> FormulaAlgorithm<T>::IterationsToConvergeCount(
    T point) const {
  return function_(point);
}

template<typename T>
AlgorithmType FormulaAlgorithm<T>::GetType() const {
  return AlgorithmType::kFormula;
}


