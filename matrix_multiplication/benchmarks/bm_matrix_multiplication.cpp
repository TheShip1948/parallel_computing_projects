#include <benchmark/benchmark.h> 
#include "matrix_multiply_brute_force.h"
#include "matrix_multiply_rows.h"
#include "matrix_multiply_cols.h"
#include "matrix_multiply_blocks.h"
#include "matrix_multiply_strassen.h"
// #include "matrix_multiply_cuda.h"
#include "matrix_multiplier_cacheoblivious.h"
#include <vector> 

static void BM_BruteForceMatrixMul(benchmark::State& state) {
    
    int N = state.range(0);
    
    // Create two N×N matrices
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));

    for (auto _ : state) {
        // Multiply the matrices
        MatrixMultiplierBruteForce multiplier_brute_force(A,B); 
        auto result = multiplier_brute_force.multiply_matrices_brute_force();

        // Prevent the compiler from optimizing out the function call
        benchmark::DoNotOptimize(result);
    }
}

// Register the function as a benchmark
BENCHMARK(BM_BruteForceMatrixMul)
    ->RangeMultiplier(2)
    ->Range(2, 2048)
    ->Complexity();



// Implement a benchmark for MatrixMultiplierRows::parallel_multiply_rows and register it with the BENCHMARK macro
static void BM_ParallelMatrixMul(benchmark::State& state) {

    int N = state.range(0);
    
    // Create two N×N matrices
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));

    for (auto _ : state) {
        // Multiply the matrices
        MatrixMultiplierRows multiplier(A, B);
        auto result = multiplier.parallel_multiply_rows(multiplier.get_optimal_thread_count(20));

        // Prevent the compiler from optimizing out the function call
        benchmark::DoNotOptimize(result);
    }
}

// Register the function as a benchmark
BENCHMARK(BM_ParallelMatrixMul)
    ->RangeMultiplier(2)
    ->Range(2, 2048)
    ->Complexity();     
    

// Implement a benchmark for MatrixMultiplierCols::parallel_multiply_cols and register it with the BENCHMARK macro
static void BM_ParallelMatrixMulCols(benchmark::State& state) {

    int N = state.range(0);
    
    // Create two N×N matrices
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));

    for (auto _ : state) {
        // Multiply the matrices
        MatrixMultiplierCols multiplier(A, B);
        auto result = multiplier.parallel_multiply_cols(multiplier.get_optimal_thread_count(20));

        // Prevent the compiler from optimizing out the function call
        benchmark::DoNotOptimize(result);
    }
}

// Register the function as a benchmark
BENCHMARK(BM_ParallelMatrixMulCols)
    ->RangeMultiplier(2)
    ->Range(2, 2048)
    ->Complexity();

// Implement a benchmark for MatrixMultiplierBlocks::parallel_multiply_blocks and register it with the BENCHMARK macro
static void BM_ParallelMatrixMulBlocks(benchmark::State& state) {

    int N = state.range(0);
    
    // Create two N×N matrices
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));

    for (auto _ : state) {
        // Multiply the matrices
        MatrixMultiplierBlocks multiplier(A, B);
        auto result = multiplier.parallel_multiply_blocks(multiplier.get_optimal_thread_count(20));

        // Prevent the compiler from optimizing out the function call
        benchmark::DoNotOptimize(result);
    }
}

// Register the function as a benchmark
BENCHMARK(BM_ParallelMatrixMulBlocks)
    ->RangeMultiplier(2)
    ->Range(2, 2048)
    ->Complexity();


static void BM_ParallelMatrixMulBlocks_future(benchmark::State& state) {

    int N = state.range(0);
    
    // Create two N×N matrices
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));

    for (auto _ : state) {
        // Multiply the matrices
        MatrixMultiplierBlocks multiplier(A, B);
        auto result = multiplier.parallel_multiply_blocks_futures(multiplier.get_optimal_thread_count(20)); 

        // Prevent the compiler from optimizing out the function call
        benchmark::DoNotOptimize(result);
    }
}

// Register the function as a benchmark
BENCHMARK(BM_ParallelMatrixMulBlocks_future)
    ->RangeMultiplier(2)
    ->Range(2, 2048)
    ->Complexity();


// Implement a benchmark for MatrixMultiplierStrassen::parallel_multiply_strassen and register it with the BENCHMARK macro
static void BM_ParallelMatrixMulStrassen(benchmark::State& state) {

    int N = state.range(0);
    
    // Create two N×N matrices
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));

    for (auto _ : state) {
        // Multiply the matrices
        auto result2_strassen = MatrixMultiplierStrassen::multiplyStrassen(A, B);
        benchmark::DoNotOptimize(result2_strassen);    
    }
}

// Register the function as a benchmark
BENCHMARK(BM_ParallelMatrixMulStrassen)
    ->RangeMultiplier(2)
    ->Range(2, 2048)
    ->Complexity();


    
// Implement a benchmark for Matrixmultipliercuda::parallel_multiply_cuda and register it with the BENCHMARK macro
// static void BM_ParallelMatrixMulCuda(benchmark::State& state) {

//     int N = state.range(0);
    
//     // Create two N×N matrices
//     std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
//     std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));

//     for (auto _ : state) {
//         // Multiply the matrices
//         auto result2_cuda = MatrixMultiplyCuda::multiply(A, B);
//         benchmark::DoNotOptimize(result2_cuda);    
//     }
// }

// // Register the function as a benchmark
// BENCHMARK(BM_ParallelMatrixMulCuda)
//     ->RangeMultiplier(2)
//     ->Range(2, 2048)
//     ->Complexity();


// Implement a benchmark for MatrixMultiplierCacheOblivious::parallel_multiply_cache_oblivious and register it with the BENCHMARK macro
static void BM_ParallelMatrixMulCacheOblivious(benchmark::State& state) {

    int N = state.range(0);
    
    // Create two N×N matrices
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));

    for (auto _ : state) {
        // Multiply the matrices
        MatrixMultiplierCacheOblivious multiplier(A, B);
        auto result = multiplier.parallel_multiply_cacheoblivious(multiplier.get_optimal_thread_count(20)); 

        // Prevent the compiler from optimizing out the function call
        benchmark::DoNotOptimize(result);
    }
}

// Register the function as a benchmark
BENCHMARK(BM_ParallelMatrixMulCacheOblivious)
    ->RangeMultiplier(2)
    ->Range(2, 2048)
    ->Complexity();

// Run the benchmark
BENCHMARK_MAIN();