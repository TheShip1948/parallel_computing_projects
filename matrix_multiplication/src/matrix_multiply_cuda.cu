#include "matrix_multiply_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

// CUDA kernel for tiled matrix multiplication
__global__ void matMulKernel(const int* A, const int* B, int* C,
                             int rowsA, int colsA, int colsB) {
    extern __shared__ int shared[];
    int* tileA = shared;
    int* tileB = shared + blockDim.x * blockDim.y;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int value = 0;

    for (int t = 0; t < (colsA + blockDim.x - 1) / blockDim.x; ++t) {
        int tiledRow = row;
        int tiledCol = t * blockDim.x + threadIdx.x;

        if (tiledRow < rowsA && tiledCol < colsA) {
            tileA[threadIdx.y * blockDim.x + threadIdx.x] =
                A[tiledRow * colsA + tiledCol];
        } else {
            tileA[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }

        tiledRow = t * blockDim.y + threadIdx.y;
        tiledCol = col;

        if (tiledRow < colsA && tiledCol < colsB) {
            tileB[threadIdx.y * blockDim.x + threadIdx.x] =
                B[tiledRow * colsB + tiledCol];
        } else {
            tileB[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }

        __syncthreads();

        for (int i = 0; i < blockDim.x; ++i) {
            value += tileA[threadIdx.y * blockDim.x + i] *
                     tileB[i * blockDim.x + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = value;
    }
}

std::vector<std::vector<int>> MatrixMultiplyCuda::multiply(
    const std::vector<std::vector<int>>& A,
    const std::vector<std::vector<int>>& B) 
{
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    // Flatten matrices
    std::vector<int> flatA(rowsA * colsA);
    std::vector<int> flatB(rowsB * colsB);
    std::vector<int> flatC(rowsA * colsB);

    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsA; ++j)
            flatA[i * colsA + j] = A[i][j];

    for (int i = 0; i < rowsB; ++i)
        for (int j = 0; j < colsB; ++j)
            flatB[i * colsB + j] = B[i][j];

    // Allocate GPU memory
    int *dA, *dB, *dC;
    cudaMalloc(&dA, flatA.size() * sizeof(int));
    cudaMalloc(&dB, flatB.size() * sizeof(int));
    cudaMalloc(&dC, flatC.size() * sizeof(int));

    cudaMemcpy(dA, flatA.data(), flatA.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, flatB.data(), flatB.size() * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x,
                  (rowsA + blockSize.y - 1) / blockSize.y);

    size_t sharedMemSize = 2 * blockSize.x * blockSize.y * sizeof(int);

    matMulKernel<<<gridSize, blockSize, sharedMemSize>>>(dA, dB, dC, rowsA, colsA, colsB);

    cudaMemcpy(flatC.data(), dC, flatC.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Convert back to 2D vector
    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB));
    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsB; ++j)
            C[i][j] = flatC[i * colsB + j];

    return C;
}
