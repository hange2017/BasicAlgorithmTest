cmake_minimum_required(VERSION 3.5)
project(chapter1_total)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fPIC")

add_executable(chapter1_total chapter1_total.cpp)
target_link_libraries(chapter1_total ${OpenCV_LIBS})

