cmake_minimum_required(VERSION 3.18)
project(MatrixMultiplication CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures (GTX 10xx series uses compute capability 61)
set(CMAKE_CUDA_ARCHITECTURES 61)

# Add executable
add_executable(matmul main.cu)

# Set CUDA flags for optimization
target_compile_options(matmul PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
    >
)

# Link CUDA runtime
target_link_libraries(matmul PRIVATE cudart)

# Print build information
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")