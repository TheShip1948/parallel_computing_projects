cmake_minimum_required(VERSION 3.18)

# We set the environment variable CUDAFLAGS to pass the vital /MD flag to the compiler 
# during the internal CMake consistency checks. This aligns the MSVC host compiler 
# with the CUDA Toolkit defaults, preventing the LNK1120 errors.
set(ENV{CUDAFLAGS} "-Xcompiler=/MD")

project(VectorMultiplication LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(vector_multiply main.cu)

# set_target_properties(vector_multiply PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES "native"
# )


set_target_properties(vector_multiply PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61"
)


# For the actual target build, we also ensure the property is set correctly (though the env var helps too)
if(WIN32)
    set_property(TARGET vector_multiply PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()