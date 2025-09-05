#include <benchmark/benchmark.h> 
#include "matrix.h"
#include <vector> 

static void BM_BruteForceMatrixMul(benchmark::State& state) {
    
    int N = state.range(0);
    
    // Create two N×N matrices
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));

    for (auto _ : state) {
        // Multiply the matrices
        auto result = multiply_matrices_brute_force(A, B);

        // Prevent the compiler from optimizing out the function call
        benchmark::DoNotOptimize(result);
    }
}

// Register the function as a benchmark
BENCHMARK(BM_BruteForceMatrixMul)
    ->RangeMultiplier(2)
    ->Range(2, 1024)
    ->Complexity();


// Run the benchmark
BENCHMARK_MAIN();