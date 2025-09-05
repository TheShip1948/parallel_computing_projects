
add_executable(benchmarks bm_matrix_multiplication.cpp)

# Link against Google Benchmark
find_package(benchmark REQUIRED)
target_link_libraries(benchmarks PRIVATE benchmark::benchmark mat_multiply)



