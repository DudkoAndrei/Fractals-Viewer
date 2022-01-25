cmake_minimum_required(VERSION 3.21)
project(fractals)

set(CMAKE_CXX_STANDARD 17)

add_executable(fractals
        main.cpp
        Image/color.cpp
        Image/gradient.cpp
        Image/image.cpp ImageProcessor/abstract_image_processor.cpp)
