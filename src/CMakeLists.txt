cmake_minimum_required(VERSION 3.12.0)
project(MatVetCgShared LANGUAGES CXX C)

include(./cmake_shared/cmake-common.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find MPI package
find_package(MPI REQUIRED)

SET(EIGEN3_DIR_LOCAL $ENV{EIGEN3_INCLUDE_DIR})        #local installation
SET(EIGEN3_DIR_PCAFRICA $ENV{mkEigenInc})             #pcafrica/mk module

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/AMSC-CodeExamples/Examples/src/
  ${CMAKE_CURRENT_SOURCE_DIR}/shared/ ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/
  ${EIGEN3_DIR_LOCAL} ${EIGEN3_DIR_PCAFRICA})
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

#Define executables
add_executable(objective1 objective1/main.cpp)
add_executable(objective2 objective2/main.cpp)
target_link_libraries(objective2 MPI::MPI_CXX)
add_executable(objective3 objective3/main.cpp)
target_link_libraries(objective3 MPI::MPI_CXX)
add_executable(mpi_matrix_test demo/mpi_matrix_test.cpp)
target_link_libraries(mpi_matrix_test MPI::MPI_CXX)
add_executable(mpi_eigen_matrix_test demo/mpi_eigen_matrix_test.cpp)
target_link_libraries(mpi_eigen_matrix_test MPI::MPI_CXX)
add_executable(mpi_eigen_sparse_matrix_test demo/mpi_eigen_sparse_matrix_test.cpp)
target_link_libraries(mpi_eigen_sparse_matrix_test MPI::MPI_CXX)
add_executable(vector_test demo/vector_test.cpp)
add_executable(eigen_structure_map_test demo/eigen_structure_map_test.cpp)
add_executable(matrix_with_vec_support_ls_test demo/matrix_with_vec_support_ls_test.cpp)
