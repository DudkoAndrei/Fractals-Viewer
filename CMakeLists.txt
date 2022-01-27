cmake_minimum_required(VERSION 3.21)
project(Fractals CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(Fractals
        Algorithms/formula_algorithm.cpp
        Controller/controller.cpp
        Cuda/cuda_formula_image_processor.cu
        Cuda/formulas_samples.cu
        Helpers/double_comparison.cpp
        Helpers/expression_parser.cpp
        Image/color.cpp
        Image/gradient.cpp
        Image/image.cpp
        ImageProcessor/abstract_image_processor.cpp
        ImageProcessor/formula_image_processor.cpp
        main.cpp
        Point/point.cpp)
