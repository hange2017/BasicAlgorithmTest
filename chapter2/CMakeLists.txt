set(CMAKE_CXX_STANDARD 14)

find_package(g2o REQUIRED)
include_directories(${g2o_INCLUDE_DIRS})

find_package(Glog REQUIRED)
include_directories(&{Glog_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(chapter2_g2o_curve_fitting chapter2_g2o_curve_fitting.cpp)
target_link_libraries(chapter2_g2o_curve_fitting g2o_core g2o_solver_dense glog g2o_stuff)

add_executable(chapter2_iter chapter2.cpp)
target_link_libraries(chapter2_iter  g2o_cli  g2o_core g2o_hierarchical  g2o_csparse_extension g2o_ext_freeglut_minimal 
    g2o_incremental  g2o_interactive g2o_interface g2o_opengl_helper g2o_parser g2o_simulator  g2o_solver_cholmod
    g2o_solver_csparse g2o_solver_dense g2o_solver_eigen g2o_solver_pcg g2o_solver_slam2d_linear g2o_solver_structure_only
    g2o_stuff g2o_types_data g2o_types_icp g2o_types_sba g2o_types_sclam2d g2o_types_sim3 g2o_types_slam2d_addons  
    g2o_types_slam3d g2o_viewer glog)#glog

add_executable(chapter2_g2o_BA chapter2_g2o_BA.cpp)
target_link_libraries(chapter2_g2o_BA ${OpenCV_LIBS} g2o_cli  g2o_core g2o_hierarchical  g2o_csparse_extension g2o_ext_freeglut_minimal 
    g2o_incremental  g2o_interactive g2o_interface g2o_opengl_helper g2o_parser g2o_simulator  g2o_solver_cholmod
    g2o_solver_csparse g2o_solver_dense g2o_solver_eigen g2o_solver_pcg g2o_solver_slam2d_linear g2o_solver_structure_only
    g2o_stuff g2o_types_data g2o_types_icp g2o_types_sba g2o_types_sclam2d g2o_types_sim3 g2o_types_slam2d_addons  
    g2o_types_slam3d g2o_viewer glog)


