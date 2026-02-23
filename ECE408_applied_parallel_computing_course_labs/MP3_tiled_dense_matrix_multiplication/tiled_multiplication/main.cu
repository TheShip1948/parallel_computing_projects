#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#define TILE_WIDTH 16

// CUDA kernel for tiled matrix multiplication
__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for (int p = 0; p < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {
        // Load tiles into shared memory
        if (Row < M && p * TILE_WIDTH + tx < N)
            ds_A[ty][tx] = A[Row * N + p * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0;

        if (p * TILE_WIDTH + ty < N && Col < K)
            ds_B[ty][tx] = B[(p * TILE_WIDTH + ty) * K + Col];
        else
            ds_B[ty][tx] = 0.0;

        __syncthreads();

        // Compute partial product
        for (int i = 0; i < TILE_WIDTH; ++i)
            Pvalue += ds_A[ty][i] * ds_B[i][tx];

        __syncthreads();
    }

    // Store result
    if (Row < M && Col < K)
        C[Row * K + Col] = Pvalue;
}

// Host function to initialize matrices
void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }
}

int main() {
    std::cout << "Starting tiled matrix multiplication" << std::endl; 
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
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
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
    
    std::cout << "\nKernel execution time (Tiled): " << milliseconds << " ms" << std::endl;
        
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