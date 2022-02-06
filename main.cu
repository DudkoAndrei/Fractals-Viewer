#include <cuda_runtime.h>
#include <iostream>

#include "Controller/controller.h"
#include "Helpers/TimeMeasurer/time_measurer.h"

int main() {
  Controller controller;

  TimeMeasurer default_run_time;
  controller.RunTest();
  std::cerr << "Default Run Time: ";
  std::cerr << default_run_time.GetElapsedSeconds() << "s\n";

  int devices_count = 0;
  cudaGetDeviceCount(&devices_count);

  if (devices_count > 0) {
    TimeMeasurer cuda_run_time;
    controller.RunCudaTest();
    std::cerr << "Cuda Run Time: ";
    std::cerr << cuda_run_time.GetElapsedSeconds() << "s\n";
  } else {
    std::cerr << "No CUDA-capable device is detected\n";
  }

  return 0;
}
