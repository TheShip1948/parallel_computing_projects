#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include <cmath>

// A heavy task to simulate work
double heavy_task(int iterations) {
    double result = 0;
    for (int i = 0; i < iterations; ++i) {
        result += std::sin(i) * std::cos(i);
    }
    return result;
}

int main() {
    const int iterations = 50000000;

    std::cout << "Comparing std::async vs std::thread performance..." << std::endl;
    std::cout << "Task: Calculate sin(i) * cos(i) for " << iterations << " iterations." << std::endl;

    // Benchmark std::thread
    auto start_thread = std::chrono::high_resolution_clock::now();
    double result_thread = 0;
    std::thread t([&result_thread, iterations]() {
        result_thread = heavy_task(iterations);
    });
    t.join();
    auto end_thread = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_thread = end_thread - start_thread;

    std::cout << "std::thread execution time: " << duration_thread.count() << " seconds" << std::endl;

    // Benchmark std::async
    auto start_async = std::chrono::high_resolution_clock::now();
    std::future<double> f = std::async(std::launch::async, heavy_task, iterations);
    double result_async = f.get();
    auto end_async = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_async = end_async - start_async;

    std::cout << "std::async execution time: " << duration_async.count() << " seconds" << std::endl;

    if (duration_async.count() < duration_thread.count()) {
        std::cout << "std::async was faster." << std::endl;
    } else {
        std::cout << "std::thread was faster." << std::endl;
    }
    
    // Prevent optimization
    if (result_thread == 12345.0) std::cout << " " << std::endl;
    if (result_async == 12345.0) std::cout << " " << std::endl;

    return 0;
}
