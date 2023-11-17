cmake_minimum_required(VERSION 3.12.0)
project(MatVetCgShared LANGUAGES CXX C)

include(../cmake_shared/cmake-common.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(vector_test vector_test.cpp)
target_include_directories(vector_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../AMSC-CodeExamples/Examples/src/
)
target_include_directories(vector_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/
)
