#pragma once

#include <chrono>
#include <string>

class TimeMeasurer {
 public:
  TimeMeasurer();
  double GetElapsedSeconds() const;

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};


