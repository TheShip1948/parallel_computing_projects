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

// Register the benchmarks with a smaller iteration count for faster execution during benchmark loop
// 1,000,000 iterations should be enough to show a difference but fast enough to run multiple times.
BENCHMARK(BM_Thread)->Arg(1000000);
BENCHMARK(BM_Async)->Arg(1000000);

BENCHMARK_MAIN();
