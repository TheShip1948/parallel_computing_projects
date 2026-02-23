#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define HIST_SIZE 256
#define BLOCK_SIZE 256

/**
 * MP7: Histogram Equalization
 * Following PMPP curriculum.
 * Uses C-style code to avoid 0xC0000409 compiler crash.
 */

__global__ void convert_to_grayscale(unsigned char* gray, float* rgb, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float r = rgb[3 * i];
        float g = rgb[3 * i + 1];
        float b = rgb[3 * i + 2];
        gray[i] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

__global__ void compute_histogram(int* hist, unsigned char* gray, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAdd(&hist[gray[i]], 1);
    }
}

__global__ void scan_naive(float* cdf, int* hist) {
    __shared__ float temp[HIST_SIZE];
    int tid = threadIdx.x;
    if (tid < HIST_SIZE) temp[tid] = (float)hist[tid];
    else temp[tid] = 0;
    __syncthreads();

    for (int stride = 1; stride < HIST_SIZE; stride *= 2) {
        float val = 0;
        if (tid >= stride) val = temp[tid - stride];
        __syncthreads();
        if (tid >= stride) temp[tid] += val;
        __syncthreads();
    }
    if (tid < HIST_SIZE) cdf[tid] = temp[tid];
}

__global__ void scan_optimized(float* cdf, int* hist) {
    __shared__ float temp[HIST_SIZE];
    int tid = threadIdx.x;
    if (tid < HIST_SIZE) temp[tid] = (float)hist[tid];
    else temp[tid] = 0;
    __syncthreads();

    for (int s = 1; s < HIST_SIZE; s *= 2) {
        int index = (tid + 1) * s * 2 - 1;
        if (index < HIST_SIZE) {
            temp[index] += temp[index - s];
        }
        __syncthreads();
    }
    for (int s = HIST_SIZE / 4; s > 0; s /= 2) {
        int index = (tid + 1) * s * 2 - 1;
        if (index + s < HIST_SIZE) {
            temp[index + s] += temp[index];
        }
        __syncthreads();
    }

    if (tid < HIST_SIZE) cdf[tid] = temp[tid];
}

__global__ void apply_equalization(float* out, float* rgb, float* cdf, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float r = rgb[3*i];
        float g = rgb[3*i+1];
        float b = rgb[3*i+2];
        unsigned char g_val = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
        float eq = (cdf[g_val] - cdf[0]) / (size - cdf[0]);
        float scale = (g_val > 0) ? (eq * 255.0f / g_val) : 0.0f;
        out[3*i]   = fminf(r * scale, 1.0f);
        out[3*i+1] = fminf(g * scale, 1.0f);
        out[3*i+2] = fminf(b * scale, 1.0f);
    }
}

int main() {
    int width = 2048, height = 2048;
    int size = width * height;
    printf("--- MP7: Histogram Equalization ---\n");

    float *d_rgb, *d_out, *d_cdf;
    unsigned char *d_gray;
    int *d_hist;
    cudaMalloc(&d_rgb, 3 * size * sizeof(float));
    cudaMalloc(&d_out, 3 * size * sizeof(float));
    cudaMalloc(&d_gray, size * sizeof(unsigned char));
    cudaMalloc(&d_hist, HIST_SIZE * sizeof(int));
    cudaMalloc(&d_cdf, HIST_SIZE * sizeof(float));

    float* h_in = (float*)malloc(3 * size * sizeof(float));
    for (int i = 0; i < 3 * size; i++) h_in[i] = (float)rand() / RAND_MAX;
    cudaMemcpy(d_rgb, h_in, 3 * size * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Iterations to make scan measurable
    const int scan_iters = 100000;

    cudaMemset(d_hist, 0, HIST_SIZE * sizeof(int));
    cudaEventRecord(start);
    convert_to_grayscale<<<blocks, BLOCK_SIZE>>>(d_gray, d_rgb, size);
    compute_histogram<<<blocks, BLOCK_SIZE>>>(d_hist, d_gray, size);
    for(int i=0; i<scan_iters; i++) scan_naive<<<1, HIST_SIZE>>>(d_cdf, d_hist);
    apply_equalization<<<blocks, BLOCK_SIZE>>>(d_out, d_rgb, d_cdf, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float t1; cudaEventElapsedTime(&t1, start, stop);
    printf("1. Naive Implementation:     %.4f ms (Scan repeated %d times)\n", t1, scan_iters);

    cudaMemset(d_hist, 0, HIST_SIZE * sizeof(int));
    cudaEventRecord(start);
    convert_to_grayscale<<<blocks, BLOCK_SIZE>>>(d_gray, d_rgb, size);
    compute_histogram<<<blocks, BLOCK_SIZE>>>(d_hist, d_gray, size);
    for(int i=0; i<scan_iters; i++) scan_optimized<<<1, HIST_SIZE>>>(d_cdf, d_hist);
    apply_equalization<<<blocks, BLOCK_SIZE>>>(d_out, d_rgb, d_cdf, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float t2; cudaEventElapsedTime(&t2, start, stop);
    printf("2. Optimized Implementation: %.4f ms (Scan repeated %d times)\n", t2, scan_iters);

    printf("Speedup: %.4fx\n", t1 / t2);

    free(h_in);
    cudaFree(d_rgb); cudaFree(d_out); cudaFree(d_gray); cudaFree(d_hist); cudaFree(d_cdf);
    return 0;
}
