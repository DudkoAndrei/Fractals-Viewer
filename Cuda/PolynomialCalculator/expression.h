#pragma once

#include <cstdio>
#include <vector>

namespace expression {
struct Segment {
  int start{0};
  size_t len{0};
};
struct AllSegments {
  Segment default_segment;
  Segment conjugate_segment;
  Segment transpose_segment;
  Segment absolute_segment;
};
}  // namespace expression

class Expression {
 public:
  Expression() = default;
  Expression(
      const std::vector<double>& default_coefs,
      const std::vector<double>& conjugate_coefs,
      const std::vector<double>& transpose_coefs,
      const std::vector<double>& absolute_coefs);

  ~Expression();

  double* GetData() const;
  int GetSize() const;

  expression::AllSegments GetSegments() const;

 private:
  double* data_{nullptr};

  expression::AllSegments segments_;
};
