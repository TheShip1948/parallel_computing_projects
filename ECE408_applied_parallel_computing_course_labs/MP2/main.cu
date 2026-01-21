#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

// CUDA kernel for matrix multiplication (no tiling)
__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

// Host function to initialize matrices
void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }
}

// Host function to verify results
bool verifyResult(float* A, float* B, float* C, int M, int N, int K) {
    const float epsilon = 1e-3;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * K + j];
            }
            if (fabs(C[i * K + j] - sum) > epsilon) {
                std::cout << "Mismatch at (" << i << "," << j << "): "
                          << "Expected " << sum << ", Got " << C[i * K + j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Matrix dimensions: A(M x N), B(N x K), C(M x K)
    int M = 1024;
    int N = 1024;
    int K = 1024;
    
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    
    // Initialize matrices
    srand(time(NULL));
    initMatrix(h_A, M, N);
    initMatrix(h_B, N, K);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    // Setup execution configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);
    
    std::cout << "Matrix dimensions: A(" << M << "x" << N << "), B(" 
              << N << "x" << K << "), C(" << M << "x" << K << ")" << std::endl;
    std::cout << "Grid dimensions: (" << gridDim.x << ", " << gridDim.y << ")" << std::endl;
    std::cout << "Block dimensions: (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch kernel and measure time
    cudaEventRecord(start);
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    
    // Wait for kernel completion
    cudaDeviceSynchronize();
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    std::cout << "\nKernel execution time: " << milliseconds << " ms" << std::endl;
    
    // Calculate GFLOPS
    // float gflops = (2.0f * M * N * K) / (milliseconds * 1e6);
    // std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
    // Verify result (only for small matrices to avoid long verification)
    if (M <= 256 && N <= 256 && K <= 256) {
        std::cout << "\nVerifying results..." << std::endl;
        if (verifyResult(h_A, h_B, h_C, M, N, K)) {
            std::cout << "Result verified: PASSED" << std::endl;
        } else {
            std::cout << "Result verified: FAILED" << std::endl;
        }
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}