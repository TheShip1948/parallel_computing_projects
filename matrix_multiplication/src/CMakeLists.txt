# CMakelists.txt

cmake_minimum_required(VERSION 3.16)

# project(mat_multiply CXX CUDA)
project(mat_multiply CXX) 

# Find CUDA package
# find_package(CUDA REQUIRED)

# Library only contains functions, no main
# add_library(mat_multiply matrix_exceptions.cpp matrix_multiply_brute_force.cpp matrix_multiply_rows.cpp matrix_multiply_cols.cpp 
#                        matrix_multiply_blocks.cpp matrix_multiply_strassen.cpp matrix_multiply_cacheoblivious.cpp matrix_multiply_cuda.cu)  

add_library(mat_multiply 
                matrix_exceptions.cpp 
                matrix_multiply_brute_force.cpp 
                matrix_multiply_rows.cpp 
                matrix_multiply_cols.cpp 
                matrix_multiply_blocks.cpp 
                matrix_multiply_strassen.cpp 
                matrix_multiplier_cacheoblivious.cpp
            )  

target_include_directories(mat_multiply PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Tell CMake to allow separable compilation for CUDA
#set_target_properties(mat_multiply PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON
#    CUDA_RESOLVE_DEVICE_SYMBOLS ON
#)

# Link CUDA runtime library to the library
#target_link_libraries(mat_multiply PUBLIC ${CUDA_LIBRARIES})

# Executable 
add_executable(main_cpp main.cpp)
target_link_libraries(main_cpp PRIVATE mat_multiply)


