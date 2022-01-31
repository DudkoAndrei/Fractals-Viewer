#include "controller.h"

#include "../ImageProcessor/formula_image_processor.h"

void Controller::RunTest() {
  image_processor_ = std::make_unique<FormulaImageProcessor>(
      Gradient{
          std::vector<Color>{
              Color{0, 0, 0},
              Color{0, 255, 255},
              Color{0, 0, 255}}});

  ImageSettings settings = {1000, 1000, 0, 0, 200, 200};
  image_processor_->GenerateImage(false, settings).WriteToFile("first.png");
}

void Controller::RunCudaTest() {
  image_processor_ = std::make_unique<FormulaImageProcessor>(
      Gradient{
          std::vector<Color>{
              Color{0, 0, 0},
              Color{0, 255, 255},
              Color{0, 0, 255}}});

  ImageSettings settings = {1000, 1000, 0, 0, 200, 200};
  image_processor_->GenerateImage(true, settings).WriteToFile("cuda_test.png");
}
