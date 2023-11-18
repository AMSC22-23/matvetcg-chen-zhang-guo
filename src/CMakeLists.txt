cmake_minimum_required(VERSION 3.12.0)
project(MatVetCgShared LANGUAGES CXX C)

include(./cmake_shared/cmake-common.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/AMSC-CodeExamples/Examples/src/
  ${CMAKE_CURRENT_SOURCE_DIR}/shared/ ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/)

# Use find_package to locate Eigen
find_package(eigen3 REQUIRED)
# Check if Eigen was found
if(EIGEN3_FOUND)
    message(STATUS "Found Eigen version ${EIGEN3_VERSION}: ${EIGEN3_INCLUDE_DIR}")
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Eigen not found. Please provide the path to Eigen manually.")
endif()

#Define executables
add_executable(objective_one objective_one/main.cpp)
add_executable(vector_test shared/vector_test.cpp)
add_executable(eigen_structure_map_test shared/eigen_structure_map_test.cpp)
add_executable(matrix_with_vec_support_ls_test shared/matrix_with_vec_support_ls_test.cpp)

