#include "expression.h"

#include <cstring>

// quite awful function
Expression::Expression(
    const std::vector<double>& default_coefs,
    const std::vector<double>& conjugate_coefs,
    const std::vector<double>& transpose_coefs,
    const std::vector<double>& absolute_coefs) {
  int new_size = default_coefs.size() +
      conjugate_coefs.size() +
      transpose_coefs.size() +
      absolute_coefs.size();
  int start_pos = 0;
  data_ = new double[new_size];

  segments_.default_segment = {start_pos, default_coefs.size()};
  if (!default_coefs.empty()) {
    memcpy(data_ + start_pos, default_coefs.data(),
           default_coefs.size() * sizeof(double));
  }
  start_pos += default_coefs.size();

  segments_.conjugate_segment = {start_pos, conjugate_coefs.size()};
  if (!conjugate_coefs.empty()) {
    memcpy(data_ + start_pos, conjugate_coefs.data(),
           conjugate_coefs.size() * sizeof(double));
  }
  start_pos += conjugate_coefs.size();

  segments_.transpose_segment = {start_pos, transpose_coefs.size()};
  if (!transpose_coefs.empty()) {
    memcpy(data_ + start_pos, transpose_coefs.data(),
           transpose_coefs.size() * sizeof(double));
  }
  start_pos += transpose_coefs.size();

  segments_.absolute_segment = {start_pos, absolute_coefs.size()};
  if (!absolute_coefs.empty()) {
    memcpy(data_ + start_pos, absolute_coefs.data(),
           absolute_coefs.size() * sizeof(double));
  }
  start_pos += absolute_coefs.size();
}

Expression::~Expression() {
  delete[] data_;
}

double* Expression::GetData() const {
  return data_;
}

int Expression::GetSize() const {
  return segments_.default_segment.len +
      segments_.conjugate_segment.len +
      segments_.transpose_segment.len +
      segments_.absolute_segment.len;
}

expression::AllSegments Expression::GetSegments() const {
  return segments_;
}
