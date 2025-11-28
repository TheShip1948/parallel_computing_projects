cmake_minimum_required(VERSION 3.10)
project(AsyncVsThread)

set(CMAKE_CXX_STANDARD 17)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable testing of the benchmark library." FORCE)

include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(async_vs_thread main.cpp)
target_link_libraries(async_vs_thread benchmark::benchmark)
