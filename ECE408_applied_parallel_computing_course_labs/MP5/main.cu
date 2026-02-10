#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <ctime>

#define BLOCK_SIZE 1024

// Implementation 1: Interleaved Addressing
// Starts from stride = 1 and then each step multiply the stride by 2
// This kernel usually suffers from branch divergence because of the modulo operation
__global__ void reduce_interleaved(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Initial load: Each thread loads 2 elements and sums them to shared memory
    // This provides initial work coarsening
    float sum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) {
        sum += g_idata[i + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Interleaved addressing reduction
    // Stride starts at 1 and doubles each iteration
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Implementation 2: Sequential Addressing
// Starts from stride = blockDim.x / 2 and goes down to 1
// This kernel avoids branch divergence by keeping active threads together
__global__ void reduce_sequential(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Initial load: Each thread loads 2 elements and sums them to shared memory
    float sum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) {
        sum += g_idata[i + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Sequential addressing reduction
    // Stride starts at half the block size and halves each iteration
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

float run_reduction(int implementation, float *d_input, float *d_output, int n, int num_blocks) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t shared_mem_size = BLOCK_SIZE * sizeof(float);

    cudaEventRecord(start);
    if (implementation == 1) {
        reduce_interleaved<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(d_input, d_output, n);
    } else {
        reduce_sequential<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(d_input, d_output, n);
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
    // Large input for better performance measurement
    const int n = 1 << 24; // 16M elements
    const size_t bytes = n * sizeof(float);

    std::cout << "--- MP5: Sum Reduction Comparison ---" << std::endl;
    std::cout << "Array Size: " << n << " (" << bytes / (1024*1024) << " MB)" << std::endl;

    // Host allocation
    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Reference CPU sum
    double cpu_sum = 0;
    for (int i = 0; i < n; i++) cpu_sum += h_input[i];

    // Device allocation
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    
    int num_blocks = (n + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);
    cudaMalloc(&d_output, num_blocks * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    std::vector<float> h_output(num_blocks);

    // Warmup
    run_reduction(1, d_input, d_output, n, num_blocks);
    run_reduction(2, d_input, d_output, n, num_blocks);

    const int iterations = 100;
    
    // Test Implementation 1
    float time1 = 0;
    for(int i=0; i<iterations; ++i) {
        time1 += run_reduction(1, d_input, d_output, n, num_blocks);
    }
    time1 /= iterations;

    // Verify Implementation 1
    cudaMemcpy(h_output.data(), d_output, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    double gpu_sum1 = 0;
    for (float val : h_output) gpu_sum1 += val;

    // Test Implementation 2
    float time2 = 0;
    for(int i=0; i<iterations; ++i) {
        time2 += run_reduction(2, d_input, d_output, n, num_blocks);
    }
    time2 /= iterations;

    // Verify Implementation 2
    cudaMemcpy(h_output.data(), d_output, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    double gpu_sum2 = 0;
    for (float val : h_output) gpu_sum2 += val;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nResults Comparison:" << std::endl;
    std::cout << "1. Interleaved (Stride 1 -> 2 -> ...): " << time1 << " ms" << std::endl;
    std::cout << "2. Sequential (Stride N/2 -> N/4 -> ...): " << time2 << " ms" << std::endl;
    
    float speedup = time1 / time2;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    std::cout << "\nVerification:" << std::endl;
    std::cout << "CPU Sum:      " << cpu_sum << std::endl;
    std::cout << "GPU Sum (1):  " << gpu_sum1 << " (Diff: " << std::abs(cpu_sum - gpu_sum1) << ")" << std::endl;
    std::cout << "GPU Sum (2):  " << gpu_sum2 << " (Diff: " << std::abs(cpu_sum - gpu_sum2) << ")" << std::endl;

    bool success = (std::abs(cpu_sum - gpu_sum1) / cpu_sum < 1e-5) && 
                   (std::abs(cpu_sum - gpu_sum2) / cpu_sum < 1e-5);
    
    if (success) {
        std::cout << "\nSUCCESS: Both implementations are correct!" << std::endl;
    } else {
        std::cout << "\nFAILURE: Sum mismatch detected." << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
