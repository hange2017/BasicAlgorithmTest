cmake_minimum_required(VERSION 3.5)
project(chapter1_feature_matching)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(chapter1)
add_subdirectory(chapter2)
add_subdirectory(chapter3)