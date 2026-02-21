#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <ctime>

#define BLOCK_SIZE 1024

// Kogge-Stone Parallel Scan (Inclusive)
// This algorithm is not work-efficient: O(N log N) additions.
// But it has a very low step complexity.
__global__ void kogge_stone_scan(float *g_idata, float *g_odata, int n) {
    __shared__ float temp[BLOCK_SIZE];
    int tid = threadIdx.x;

    // Load data from global to shared memory
    if (tid < n) {
        temp[tid] = g_idata[tid];
    } else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    // Kogge-Stone Iterations
    for (int stride = 1; stride < n; stride *= 2) {
        float val = 0.0f;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write back to global memory
    if (tid < n) {
        g_odata[tid] = temp[tid];
    }
}

// Brent-Kung Parallel Scan (Inclusive)
// This algorithm is work-efficient: O(N) additions.
// Phase 1: Reduction (Up-sweep)
// Phase 2: Post-scan (Down-sweep)
__global__ void brent_kung_scan(float *g_idata, float *g_odata, int n) {
    __shared__ float temp[BLOCK_SIZE];
    int tid = threadIdx.x;

    // Load data: Each thread can load multiple elements to achieve work efficiency,
    // but for simplicity in comparing with Kogge-Stone, we use one element per thread
    // and focus on the tree structure.
    if (tid < n) {
        temp[tid] = g_idata[tid];
    } else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    // Phase 1: Reduction (Up-sweep)
    for (int stride = 1; stride < n; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Phase 2: Post-scan (Down-sweep)
    for (int stride = n / 4; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < n) {
            temp[index + stride] += temp[index];
        }
        __syncthreads();
    }

    // Write back
    if (tid < n) {
        g_odata[tid] = temp[tid];
    }
}

void cpu_scan(const float *input, float *output, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

float run_scan(int mode, float *d_input, float *d_output, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (mode == 0) {
        kogge_stone_scan<<<1, BLOCK_SIZE>>>(d_input, d_output, n);
    } else {
        brent_kung_scan<<<1, BLOCK_SIZE>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

int main() {
    const int n = BLOCK_SIZE;
    const size_t bytes = n * sizeof(float);

    std::cout << "--- MP6: Prefix Sum (Scan) Comparison ---" << std::endl;
    std::cout << "Array Size: " << n << std::endl;

    // Host allocation
    std::vector<float> h_input(n);
    std::vector<float> h_output_cpu(n);
    std::vector<float> h_output_gpu(n);

    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f; // Simplified for easier verification
    }

    // CPU Reference
    cpu_scan(h_input.data(), h_output_cpu.data(), n);

    // Device allocation
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // Warmup
    run_scan(0, d_input, d_output, n);
    run_scan(1, d_input, d_output, n);

    const int iterations = 1000;

    // Test Kogge-Stone
    float time_ks = 0;
    for (int i = 0; i < iterations; i++) {
        time_ks += run_scan(0, d_input, d_output, n);
    }
    time_ks /= iterations;

    // Verify Kogge-Stone
    cudaMemcpy(h_output_gpu.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    bool ks_correct = true;
    for (int i = 0; i < n; i++) {
        if (std::abs(h_output_gpu[i] - h_output_cpu[i]) > 1e-4) {
            ks_correct = false;
            break;
        }
    }

    // Test Brent-Kung
    float time_bk = 0;
    for (int i = 0; i < iterations; i++) {
        time_bk += run_scan(1, d_input, d_output, n);
    }
    time_bk /= iterations;

    // Verify Brent-Kung
    cudaMemcpy(h_output_gpu.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    bool bk_correct = true;
    for (int i = 0; i < n; i++) {
        if (std::abs(h_output_gpu[i] - h_output_cpu[i]) > 1e-4) {
            bk_correct = false;
            break;
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nResults Comparison (over " << iterations << " iterations):" << std::endl;
    std::cout << "1. Kogge-Stone: " << time_ks << " ms (" << (ks_correct ? "CORRECT" : "FAILED") << ")" << std::endl;
    std::cout << "2. Brent-Kung:  " << time_bk << " ms (" << (bk_correct ? "CORRECT" : "FAILED") << ")" << std::endl;

    float speedup = time_ks / time_bk;
    std::cout << "Execution Time Ratio (KS / BK): " << speedup << "x" << std::endl;

    std::cout << "\nWork Efficiency Analysis:" << std::endl;
    std::cout << "Kogge-Stone Work (Approximation): N log2(N) = " << n << " * " << log2(n) << " = " << n * log2(n) << " additions." << std::endl;
    std::cout << "Brent-Kung Work (Approximation): 2 * (N - 1) = " << 2 * (n - 1) << " additions." << std::endl;
    
    float work_ratio = (n * log2(n)) / (2.0f * (n - 1));
    std::cout << "Work Ratio (KS / BK): " << work_ratio << "x" << std::endl;

    if (ks_correct && bk_correct) {
        std::cout << "\nSUCCESS: Both implementations are correct!" << std::endl;
    } else {
        std::cout << "\nFAILURE: One or both implementations failed verification." << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
