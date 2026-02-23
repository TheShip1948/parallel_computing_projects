cmake_minimum_required(VERSION 3.18)
project(MP7 CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 61)

add_executable(MP7 main.cu)

target_compile_options(MP7 PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
    >
)

target_link_libraries(MP7 PRIVATE cudart)
