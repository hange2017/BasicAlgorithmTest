cmake_minimum_required(VERSION 3.5)
project(chapter1_feature_matching)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fPIC")

add_executable(chapter1_feature_matching chapter1_feature_matching.cpp)
target_link_libraries(chapter1_feature_matching ${OpenCV_LIBS})

