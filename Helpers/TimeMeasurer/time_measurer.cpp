#include "time_measurer.h"

TimeMeasurer::TimeMeasurer() :
    start_(std::chrono::high_resolution_clock::now()) {}

double TimeMeasurer::GetElapsedSeconds() const {
  auto now = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = now - start_;
  return duration.count();
}
