cmake_minimum_required(VERSION 3.18)

project(VectorMultiplication LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(vector_multiply main.cu)

set_target_properties(vector_multiply PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)

# Ensure we use the dynamic runtime to match standard C++ builds on Windows
if(WIN32)
    set_property(TARGET vector_multiply PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()