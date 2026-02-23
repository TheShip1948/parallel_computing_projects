#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

// CUDA kernel for row-column shared memory matrix multiplication
__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int N, int K) {
    // Shared memory for row of A and column of B
    // Dynamically allocated: size 2 * N * sizeof(float)
    extern __shared__ float shared_mem[];
    float* ds_A = shared_mem;         // size N
    float* ds_B = &shared_mem[N];      // size N

    int row = blockIdx.y; // Row index i
    int col = blockIdx.x; // Column index j
    int tx = threadIdx.x;

    // Collaborative load of row A[row] and column B[col]
    for (int k = tx; k < N; k += blockDim.x) {
        ds_A[k] = A[row * N + k];
        ds_B[k] = B[k * K + col];
    }
    __syncthreads();

    // Dot product with parallel reduction
    float partialSum = 0;
    for (int k = tx; k < N; k += blockDim.x) {
        partialSum += ds_A[k] * ds_B[k];
    }

    // Reuse part of shared memory for reduction within the block
    __syncthreads(); 
    ds_A[tx] = partialSum;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            ds_A[tx] += ds_A[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0) {
        C[row * K + col] = ds_A[0];
    }
}

// Host function to initialize matrices
void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }
}

int main() {
    std::cout << "Starting row-column shared memory matrix multiplication" << std::endl; 
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
    // Each block computes one element (row, col) of C
    // Block size is 256 threads for reduction
    int threadsPerBlock = 256;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(K, M);
    
    // Calculate required shared memory: one row of A (N) and one col of B (N)
    size_t sharedMemSize = 2 * N * sizeof(float);
    
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
    matrixMulKernel<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, M, N, K);
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
    
    std::cout << "\nKernel execution time (Row-Col Shared): " << milliseconds << " ms" << std::endl;
        
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