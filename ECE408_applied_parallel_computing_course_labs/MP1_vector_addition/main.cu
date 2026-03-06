
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void vecMul(const float *in1, const float *in2, float *out, int len) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < len) {
        out[i] = in1[i] * in2[i];
    }
}

void vecMulHost(const float *in1, const float *in2, float *out, int len) {
    for (int i = 0; i < len; ++i) {
        out[i] = in1[i] * in2[i];
    }
}

int main(int argc, char **argv) {
    int inputLength = 1024; // Default size
    if (argc > 1) inputLength = atoi(argv[1]);

    size_t bytes = inputLength * sizeof(float);

    printf("Vector Multiplication with size %d\n", inputLength);

    // Allocate host memory
    float* hostInput1 = (float*)malloc(bytes);
    float* hostInput2 = (float*)malloc(bytes);
    float* hostOutput = (float*)malloc(bytes);
    float* expectedOutput = (float*)malloc(bytes);

    // Initialize data
    for (int i = 0; i < inputLength; ++i) {
        hostInput1[i] = static_cast<float>(rand()) / RAND_MAX;
        hostInput2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *deviceInput1, *deviceInput2, *deviceOutput;
    cudaMalloc((void **)&deviceInput1, bytes);
    cudaMalloc((void **)&deviceInput2, bytes);
    cudaMalloc((void **)&deviceOutput, bytes);

    // Copy data to GPU
    cudaMemcpy(deviceInput1, hostInput1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (inputLength + blockSize - 1) / blockSize;
    
    vecMul<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(hostOutput, deviceOutput, bytes, cudaMemcpyDeviceToHost);

    // Compute reference
    vecMulHost(hostInput1, hostInput2, expectedOutput, inputLength);

    // Verify
    bool match = true;
    for (int i = 0; i < inputLength; ++i) {
        if (fabs(hostOutput[i] - expectedOutput[i]) > 1e-5) {
            printf("Mismatch at %d: %f != %f\n", i, hostOutput[i], expectedOutput[i]);
            match = false;
            break;
        }
    }

    if (match) {
        printf("Test Passed!\n");
    } else {
        printf("Test Failed!\n");
    }

    // Free memory
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(expectedOutput);

    return 0;
}