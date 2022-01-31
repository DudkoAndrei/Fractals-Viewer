#include <iostream>

#include "Controller/controller.h"
#include "Helpers/TimeMeasurer/time_measurer.h"

int main() {
  Controller controller;

  TimeMeasurer default_run_time;
  controller.RunTest();
  std::cerr << "Default Run Time: ";
  std::cerr << default_run_time.GetElapsedSeconds() << "s\n";
  TimeMeasurer cuda_run_time;
  controller.RunCudaTest();
  std::cerr << "Cuda Run Time: ";
  std::cerr << cuda_run_time.GetElapsedSeconds() << "s\n";
  return 0;
}
