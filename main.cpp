#include "Controller/controller.h"

int main() {
  Controller controller;

  controller.RunTest();
  controller.RunCudaTest();
  return 0;
}
