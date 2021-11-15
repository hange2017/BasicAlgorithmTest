cmake_minimum_required(VERSION 3.5)
project(chapter1_feature_matching)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fPIC")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(chapter1_feature_matching chapter1_feature_matching.cpp)
target_link_libraries(chapter1_feature_matching ${penCV_LIBS})

