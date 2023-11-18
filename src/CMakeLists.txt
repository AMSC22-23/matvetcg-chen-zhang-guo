cmake_minimum_required(VERSION 3.12.0)
project(MatVetCgShared LANGUAGES CXX C)

include(./cmake_shared/cmake-common.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(objective_one objective_one/main.cpp shared/MatrixWithVecSupport.cpp)
target_include_directories(objective_one PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/AMSC-CodeExamples/Examples/src/
)
target_include_directories(objective_one PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/shared/
)
target_include_directories(objective_one PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/
)

add_executable(vector_test shared/vector_test.cpp)
target_include_directories(vector_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/AMSC-CodeExamples/Examples/src/
)
target_include_directories(vector_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/shared/
)

add_executable(eigen_structure_map_test shared/eigen_structure_map_test.cpp)
target_include_directories(eigen_structure_map_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/AMSC-CodeExamples/Examples/src/
)
target_include_directories(eigen_structure_map_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/shared/
)

# Use find_package to locate Eigen
find_package(eigen3 REQUIRED)

# Check if Eigen was found
if(EIGEN3_FOUND)
    message(STATUS "Found Eigen version ${EIGEN3_VERSION}: ${EIGEN3_INCLUDE_DIR}")
    
    target_include_directories(objective_one PRIVATE
        ${EIGEN3_INCLUDE_DIR}
    )
    target_include_directories(eigen_structure_map_test PRIVATE
        ${EIGEN3_INCLUDE_DIR}
    )
    
else()
    message(FATAL_ERROR "Eigen not found. Please provide the path to Eigen manually.")
endif()
