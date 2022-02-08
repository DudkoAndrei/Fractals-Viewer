cmake_minimum_required(VERSION 3.21)
project(Fractals CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,leak,undefined")

add_executable(Fractals
        Algorithms/formula_algorithm.cpp
        Controller/controller.cpp
        Cuda/cuda_error_handler.cu
        Cuda/cuda_formula_algorithm.cu
        Helpers/double_comparison.cpp
        Image/color.cpp
        Image/gradient.cpp
        Image/image.cpp
        Image/ImageProcessor/formula_image_processor.cpp
        main.cu
        Point/point.cpp
        Helpers/TimeMeasurer/time_measurer.cpp
        Cuda/PolynomialCalculator/expression.cpp)
