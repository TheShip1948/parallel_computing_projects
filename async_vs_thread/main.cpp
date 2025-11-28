#include <benchmark/benchmark.h>
#include <thread>
#include <future>
#include <cmath>
#include <vector>

// A heavy task to simulate work
double heavy_task(int64_t iterations) {
    double result = 0;
    for (int i = 0; i < iterations; ++i) {
        result += std::sin(i) * std::cos(i);
    }
    return result;
}

static void BM_Thread(benchmark::State& state) {
    const int64_t iterations = state.range(0);
    for (auto _ : state) {
        double result_thread = 0;
        std::thread t([&result_thread, iterations]() {
            result_thread = heavy_task(iterations);
        });
        t.join();
        benchmark::DoNotOptimize(result_thread);
    }
}

static void BM_Async(benchmark::State& state) {
    const int64_t iterations = state.range(0);
    for (auto _ : state) {
        std::future<double> f = std::async(std::launch::async, heavy_task, iterations);
        double result = f.get();
        benchmark::DoNotOptimize(result);
    }
}

// Register the benchmarks with a range of iterations from 2 to 256
BENCHMARK(BM_Thread)->Range(2, 256)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Async)->Range(2, 256)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
