#include "controller.h"

std::optional<long long> MandelbrotSet(const Point& point) {
  std::complex<double> c((point.x() - 500) / 200.0,
                         (point.y() - 500) / 200.0);
  std::complex<double> z(0.0, 0.0);

  size_t iteration = 0;
  while (iteration < 1000 && abs(z) < (2 << 8)) {
    z = (z * z) + c;
    ++iteration;
  }

  if (iteration < 1000) {
    return {iteration};
  }
  return std::nullopt;
}

void Controller::RunTest() {
  image_processor_ = std::make_unique<FormulaImageProcessor>();

  ImageSettings settings = {1000, 1000, 0, 0, 1, 1};
  image_processor_->GenerateImage(false,settings).WriteToFile("first.png");
}

void Controller::RunCudaTest() {
  image_processor_ = std::make_unique<FormulaImageProcessor>();

  ImageSettings settings = {1000, 1000, 0, 0, 200, 200};
  image_processor_->GenerateImage(true,settings).WriteToFile("cuda_test.png");
}
