cmake_minimum_required(VERSION 3.12.0)
project(MatVetCgObjOne LANGUAGES CXX C)

include(../cmake_shared/cmake-common.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main main.cpp)
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../AMSC-CodeExamples/Examples/src/
)
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared/
)
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../algorithms/
)

# # Use find_package to locate Eigen
# find_package(eigen3 REQUIRED)

# # Check if Eigen was found
# if(EIGEN3_FOUND)
#     message(STATUS "Found Eigen version ${EIGEN3_VERSION}: ${EIGEN3_INCLUDE_DIR}")
#     
#     target_include_directories(main PRIVATE
#         ${EIGEN3_INCLUDE_DIR}
#     )
#     
# else()
#     message(FATAL_ERROR "Eigen not found. Please provide the path to Eigen manually.")
# endif()
