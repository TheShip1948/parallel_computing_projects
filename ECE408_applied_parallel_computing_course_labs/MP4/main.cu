#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

// Constants for convolution
#define MASK_WIDTH 5
#define MASK_RADIUS (MASK_WIDTH / 2)
#define BLOCK_WIDTH 16
#define O_TILE_WIDTH (BLOCK_WIDTH - (MASK_WIDTH - 1))

// Mask stored in constant memory for performance
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH];

// 2D Tiled Convolution Kernel
// This implementation follows the strategy from "Programming Massively Parallel Processors"
// Each block loads an input tile (BLOCK_WIDTH x BLOCK_WIDTH) into shared memory.
// Threads within the output tile boundary (O_TILE_WIDTH x O_TILE_WIDTH) compute the convolution.
__global__ void convolution_2D_tiled_kernel(float *I, float *O, int width, int height) {
    __shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate global indices for loading input
    // The input tile is shifted by MASK_RADIUS to include ghost cells
    int row_i = blockIdx.y * O_TILE_WIDTH + ty - MASK_RADIUS;
    int col_i = blockIdx.x * O_TILE_WIDTH + tx - MASK_RADIUS;

    // Load elements into shared memory, handle boundary conditions with zeros (clamped to zero)
    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
        N_ds[ty][tx] = I[row_i * width + col_i];
    } else {
        N_ds[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Calculate global indices for output
    int row_o = blockIdx.y * O_TILE_WIDTH + ty;
    int col_o = blockIdx.x * O_TILE_WIDTH + tx;

    // Only threads that are within the valid output tile compute the result
    // These are the inner threads that have all their needed mask neighbors in shared memory
    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
        float Pvalue = 0.0f;
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                // Neighbors are conveniently available in shared memory with indices offset by (ty, tx)
                Pvalue += Mc[i][j] * N_ds[ty + i][tx + j];
            }
        }
        
        // Final boundary check for the output image dimensions
        if (row_o < height && col_o < width) {
            O[row_o * width + col_o] = Pvalue;
        }
    }
}

// CPU Implementation for result verification
void convolution_2D_cpu(float *I, float *mask, float *O, int width, int height) {
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            float sum = 0.0f;
            for (int i = 0; i < MASK_WIDTH; i++) {
                for (int j = 0; j < MASK_WIDTH; j++) {
                    int row_i = r + i - MASK_RADIUS;
                    int col_i = c + j - MASK_RADIUS;
                    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
                        sum += I[row_i * width + col_i] * mask[i * MASK_WIDTH + j];
                    }
                }
            }
            O[r * width + c] = sum;
        }
    }
}

int main() {
    // Large image for testing performance and correctness
    int width = 1024;
    int height = 1024;
    
    size_t size = width * height * sizeof(float);
    size_t maskSize = MASK_WIDTH * MASK_WIDTH * sizeof(float);

    // Host memory allocation
    float *h_I = (float*)malloc(size);
    float *h_O = (float*)malloc(size);
    float *h_mask = (float*)malloc(maskSize);
    float *h_O_verify = (float*)malloc(size);

    // Initialize host data with random values
    srand(static_cast<unsigned int>(time(NULL)));
    for (int i = 0; i < width * height; i++) {
        h_I[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++) {
        h_mask[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device memory allocation
    float *d_I, *d_O;
    if (cudaMalloc(&d_I, size) != cudaSuccess || cudaMalloc(&d_O, size) != cudaSuccess) {
        std::cerr << "CUDA Memory Allocation failed!" << std::endl;
        return -1;
    }

    // Copy input image to device global memory
    cudaMemcpy(d_I, h_I, size, cudaMemcpyHostToDevice);
    
    // Copy mask to constant memory
    cudaMemcpyToSymbol(Mc, h_mask, maskSize);

    // Execution configuration
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
    // The number of blocks must cover the entire image using O_TILE_WIDTH output elements each
    dim3 gridDim((width + O_TILE_WIDTH - 1) / O_TILE_WIDTH, 
                 (height + O_TILE_WIDTH - 1) / O_TILE_WIDTH);

    std::cout << "--- MP4: 2D Tiled Convolution ---" << std::endl;
    std::cout << "Image Dimensions: " << width << " x " << height << std::endl;
    std::cout << "Mask Dimensions: " << MASK_WIDTH << " x " << MASK_WIDTH << " (Radius: " << MASK_RADIUS << ")" << std::endl;
    std::cout << "Block Dimensions: " << BLOCK_WIDTH << " x " << BLOCK_WIDTH << std::endl;
    std::cout << "Output Tile Width: " << O_TILE_WIDTH << std::endl;
    std::cout << "Grid Dimensions: " << gridDim.x << " x " << gridDim.y << std::endl;

    // Setup timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel execution
    cudaEventRecord(start);
    convolution_2D_tiled_kernel<<<gridDim, blockDim>>>(d_I, d_O, width, height);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after kernel launch: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy result back to host
    cudaMemcpy(h_O, d_O, size, cudaMemcpyDeviceToHost);

    // Verification against CPU
    std::cout << "Verifying results against CPU implementation..." << std::endl;
    convolution_2D_cpu(h_I, h_mask, h_O_verify, width, height);

    double maxError = 0.0;
    for (int i = 0; i < width * height; i++) {
        double diff = std::abs(h_O[i] - h_O_verify[i]);
        if (diff > maxError) maxError = diff;
    }
    
    std::cout << "Verification Conclusion: " << std::endl;
    std::cout << "Max Absolute Error: " << maxError << std::endl;
    if (maxError < 1e-4) {
        std::cout << "SUCCESS: Results match CPU output!" << std::endl;
    } else {
        std::cout << "FAILURE: Results deviate too much from CPU output." << std::endl;
    }

    // Cleanup
    cudaFree(d_I);
    cudaFree(d_O);
    free(h_I);
    free(h_O);
    free(h_mask);
    free(h_O_verify);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
