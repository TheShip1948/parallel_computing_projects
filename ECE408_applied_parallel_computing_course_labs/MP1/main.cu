
#include <iostream>
#include <vector>
#include <cmath>
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

    std::cout << "Vector Multiplication with size " << inputLength << std::endl;

    // Allocate host memory
    std::vector<float> hostInput1(inputLength);
    std::vector<float> hostInput2(inputLength);
    std::vector<float> hostOutput(inputLength);
    std::vector<float> expectedOutput(inputLength);

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
    cudaMemcpy(deviceInput1, hostInput1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2.data(), bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (inputLength + blockSize - 1) / blockSize;
    
    vecMul<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(hostOutput.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost);

    // Compute reference
    vecMulHost(hostInput1.data(), hostInput2.data(), expectedOutput.data(), inputLength);

    // Verify
    bool match = true;
    for (int i = 0; i < inputLength; ++i) {
        if (std::abs(hostOutput[i] - expectedOutput[i]) > 1e-5) {
            std::cout << "Mismatch at " << i << ": " << hostOutput[i] << " != " << expectedOutput[i] << std::endl;
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "Test Passed!" << std::endl;
    } else {
        std::cout << "Test Failed!" << std::endl;
    }

    // Free memory
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    return 0;
}
