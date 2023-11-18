cmake_minimum_required(VERSION 3.12.0)
project(MatVetCgShared LANGUAGES CXX C)

include(./cmake_shared/cmake-common.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

SET(EIGEN3_DIR_LOCAL $ENV{EIGEN3_INCLUDE_DIR})        #local installation
SET(EIGEN3_DIR_PCAFRICA $ENV{mkEigenInc})             #pcafrica/mk module
SET(EIGEN3_DIR_GITHUB_ACTIONS "/usr/include/eigen3")  #Github actions server

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/AMSC-CodeExamples/Examples/src/
  ${CMAKE_CURRENT_SOURCE_DIR}/shared/ ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/
  ${EIGEN3_DIR_LOCAL} ${EIGEN3_DIR_PCAFRICA} ${EIGEN3_DIR_GITHUB_ACTIONS})

#Define executables
add_executable(objective_one objective_one/main.cpp)
add_executable(vector_test shared/vector_test.cpp)
add_executable(eigen_structure_map_test shared/eigen_structure_map_test.cpp)
add_executable(matrix_with_vec_support_ls_test shared/matrix_with_vec_support_ls_test.cpp)
