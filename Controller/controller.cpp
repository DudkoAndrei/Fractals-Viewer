#include "controller.h"

#include "../ImageProcessor/formula_image_processor.h"

void Controller::RunTest() {
  image_processor_ = std::make_unique<FormulaImageProcessor>();

  ImageSettings settings = {1000, 1000, 0, 0, 1, 1};
  image_processor_->GenerateImage(false,settings).WriteToFile("first.png");
}


void Controller::RunCudaTest() {
  image_processor_ = std::make_unique<FormulaImageProcessor>();

  ImageSettings settings = {1000, 1000, 0, 0, 200, 200};
  image_processor_->GenerateImage(true, settings).WriteToFile("cuda_test.png");
}
