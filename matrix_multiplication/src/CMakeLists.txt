# CMakelists.txt

cmake_minimum_required(VERSION 3.16)

project(mat_multiply CXX)

# Library only contains functions, no main
add_library(mat_multiply matrix.cpp) 
target_include_directories(mat_multiply PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Executable 
add_executable(main_cpp main.cpp)
target_link_libraries(main_cpp PRIVATE mat_multiply)


