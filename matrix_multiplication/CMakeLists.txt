cmake_minimum_required(VERSION 3.16)
project(MatrixMul CXX)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src)         # defines the mat_multiply library
add_subdirectory(benchmarks)  # defines the benchmarks executable