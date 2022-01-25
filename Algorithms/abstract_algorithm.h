#pragma once

enum class AlgorithmType {
  kNone,
  kFormula,
  kRandomized,
};

class AbstractAlgorithm {
 public:
  virtual AlgorithmType GetType() const = 0;
};
