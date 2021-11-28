cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${pcl_INCLUDE_DIRS})

add_executable(check_rslt check_rslt.cpp)
target_link_libraries(check_rslt ${PCL_LIBRARIES} )
