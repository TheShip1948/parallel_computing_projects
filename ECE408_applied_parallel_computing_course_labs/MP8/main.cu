#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/**
 * MP8: Sparse Matrix-Vector Multiplication (SpMV)
 * Following Programming Massively Parallel Processors (PMPP) Chapter 9.
 * Comparison between CSR (Baseline) and JDS (Optimized) formats.
 * 
 * Simplified C-style implementation to avoid 0xC0000409 compiler crash.
 */

#define BLOCK_SIZE 256

// --- CSR (Compressed Sparse Row) Format ---
typedef struct {
    int num_rows;
    int num_cols;
    int num_nonzeros;
    float* data;
    int* col_index;
    int* row_ptr;
} CSRMatrix;

// --- JDS (Jagged Diagonal Storage) Format ---
typedef struct {
    int num_rows;
    int num_cols;
    int num_nonzeros;
    int num_jagged_diagonals;
    float* data;
    int* col_index;
    int* iteration_ptr;
    int* perm;
} JDSMatrix;

// --- Kernels ---

/**
 * CSR SpMV Kernel
 * Each thread handles one row.
 */
__global__ void spmv_csr_kernel(int num_rows, const int* ptr, const int* indices, const float* data, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        int row_start = ptr[row];
        int row_end = ptr[row + 1];
        for (int i = row_start; i < row_end; i++) {
            dot += data[i] * x[indices[i]];
        }
        y[row] = dot;
    }
}

/**
 * JDS SpMV Kernel (Transposed)
 * Each thread handles one reordered row.
 */
__global__ void spmv_jds_kernel(int num_rows, int num_jagged_diagonals, const int* iteration_ptr, const int* indices, const float* data, const int* perm, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        for (int j = 0; j < num_jagged_diagonals; j++) {
            int diag_start = iteration_ptr[j];
            int diag_end = iteration_ptr[j+1];
            int row_count_in_diag = diag_end - diag_start;
            
            if (row < row_count_in_diag) {
                int pos = diag_start + row;
                dot += data[pos] * x[indices[pos]];
            }
        }
        y[perm[row]] = dot;
    }
}

// --- Utilities ---

void spmv_cpu(CSRMatrix mat, const float* x, float* y) {
    for (int i = 0; i < mat.num_rows; i++) {
        float dot = 0.0f;
        for (int j = mat.row_ptr[i]; j < mat.row_ptr[i+1]; j++) {
            dot += mat.data[j] * x[mat.col_index[j]];
        }
        y[i] = dot;
    }
}

typedef struct {
    int original_index;
    int length;
} RowInfo;

int compare_row_info(const void* a, const void* b) {
    return ((RowInfo*)b)->length - ((RowInfo*)a)->length;
}

CSRMatrix generate_random_sparse_matrix(int rows, int cols, float density) {
    CSRMatrix mat;
    mat.num_rows = rows;
    mat.num_cols = cols;
    mat.row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    mat.row_ptr[0] = 0;
    
    // Preliminary pass to count non-zeros
    int estimated_nz = (int)(rows * cols * density * 1.5f);
    mat.data = (float*)malloc(estimated_nz * sizeof(float));
    mat.col_index = (int*)malloc(estimated_nz * sizeof(int));
    
    int nz = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if ((float)rand() / RAND_MAX < density) {
                if (nz >= estimated_nz) {
                    estimated_nz *= 2;
                    mat.data = (float*)realloc(mat.data, estimated_nz * sizeof(float));
                    mat.col_index = (int*)realloc(mat.col_index, estimated_nz * sizeof(int));
                }
                mat.data[nz] = (float)rand() / RAND_MAX;
                mat.col_index[nz] = j;
                nz++;
            }
        }
        mat.row_ptr[i + 1] = nz;
    }
    mat.num_nonzeros = nz;
    return mat;
}

JDSMatrix convert_to_jds(CSRMatrix csr) {
    JDSMatrix jds;
    jds.num_rows = csr.num_rows;
    jds.num_cols = csr.num_cols;
    jds.num_nonzeros = csr.num_nonzeros;

    RowInfo* rows = (RowInfo*)malloc(csr.num_rows * sizeof(RowInfo));
    for (int i = 0; i < csr.num_rows; i++) {
        rows[i].original_index = i;
        rows[i].length = csr.row_ptr[i + 1] - csr.row_ptr[i];
    }

    qsort(rows, csr.num_rows, sizeof(RowInfo), compare_row_info);

    jds.perm = (int*)malloc(csr.num_rows * sizeof(int));
    for (int i = 0; i < csr.num_rows; i++) {
        jds.perm[i] = rows[i].original_index;
    }

    int max_len = (csr.num_rows > 0) ? rows[0].length : 0;
    jds.num_jagged_diagonals = max_len;
    jds.iteration_ptr = (int*)malloc((max_len + 1) * sizeof(int));

    jds.data = (float*)malloc(csr.num_nonzeros * sizeof(float));
    jds.col_index = (int*)malloc(csr.num_nonzeros * sizeof(int));

    int current_pos = 0;
    for (int j = 0; j < max_len; j++) {
        jds.iteration_ptr[j] = current_pos;
        for (int i = 0; i < csr.num_rows; i++) {
            if (rows[i].length > j) {
                int csr_row = rows[i].original_index;
                int csr_pos = csr.row_ptr[csr_row] + j;
                jds.data[current_pos] = csr.data[csr_pos];
                jds.col_index[current_pos] = csr.col_index[csr_pos];
                current_pos++;
            }
        }
    }
    jds.iteration_ptr[max_len] = current_pos;

    free(rows);
    return jds;
}

void free_csr(CSRMatrix m) {
    free(m.data); free(m.col_index); free(m.row_ptr);
}

void free_jds(JDSMatrix m) {
    free(m.data); free(m.col_index); free(m.iteration_ptr); free(m.perm);
}

int main() {
    int num_rows = 2048; // Slightly smaller to be safe
    int num_cols = 2048;
    float density = 0.05f;

    printf("--- MP8: Sparse Matrix-Vector Multiplication (SpMV) Comparison ---\n");
    printf("Matrix: %d x %d (Density: %.2f%%)\n", num_rows, num_cols, density * 100.0f);

    CSRMatrix csr = generate_random_sparse_matrix(num_rows, num_cols, density);
    JDSMatrix jds = convert_to_jds(csr);
    
    float* h_x = (float*)malloc(num_cols * sizeof(float));
    for (int i = 0; i < num_cols; i++) h_x[i] = (float)rand() / RAND_MAX;
    
    float* h_y_cpu = (float*)malloc(num_rows * sizeof(float));
    spmv_cpu(csr, h_x, h_y_cpu);

    float *d_csr_data, *d_x, *d_y, *d_jds_data;
    int *d_csr_ptr, *d_csr_indices, *d_jds_indices, *d_jds_iter_ptr, *d_jds_perm;

    cudaMalloc(&d_csr_data, csr.num_nonzeros * sizeof(float));
    cudaMalloc(&d_csr_indices, csr.num_nonzeros * sizeof(int));
    cudaMalloc(&d_csr_ptr, (csr.num_rows + 1) * sizeof(int));
    
    cudaMalloc(&d_jds_data, jds.num_nonzeros * sizeof(float));
    cudaMalloc(&d_jds_indices, jds.num_nonzeros * sizeof(int));
    cudaMalloc(&d_jds_iter_ptr, (jds.num_jagged_diagonals + 1) * sizeof(int));
    cudaMalloc(&d_jds_perm, jds.num_rows * sizeof(int));

    cudaMalloc(&d_x, num_cols * sizeof(float));
    cudaMalloc(&d_y, num_rows * sizeof(float));

    cudaMemcpy(d_csr_data, csr.data, csr.num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_indices, csr.col_index, csr.num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_ptr, csr.row_ptr, (csr.num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_jds_data, jds.data, jds.num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jds_indices, jds.col_index, jds.num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jds_iter_ptr, jds.iteration_ptr, (jds.num_jagged_diagonals + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jds_perm, jds.perm, jds.num_rows * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iterations = 1000;
    int blocks = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // CSR
    cudaMemset(d_y, 0, num_rows * sizeof(float));
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        spmv_csr_kernel<<<blocks, BLOCK_SIZE>>>(num_rows, d_csr_ptr, d_csr_indices, d_csr_data, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float t_csr;
    cudaEventElapsedTime(&t_csr, start, stop);
    t_csr /= iterations;

    float* h_y_gpu = (float*)malloc(num_rows * sizeof(float));
    cudaMemcpy(h_y_gpu, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    int csr_ok = 1;
    for (int i = 0; i < num_rows; i++) if (fabs(h_y_gpu[i] - h_y_cpu[i]) > 1e-3) { csr_ok = 0; break; }

    // JDS
    cudaMemset(d_y, 0, num_rows * sizeof(float));
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        spmv_jds_kernel<<<blocks, BLOCK_SIZE>>>(num_rows, jds.num_jagged_diagonals, d_jds_iter_ptr, d_jds_indices, d_jds_data, d_jds_perm, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float t_jds;
    cudaEventElapsedTime(&t_jds, start, stop);
    t_jds /= iterations;

    cudaMemcpy(h_y_gpu, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    int jds_ok = 1;
    for (int i = 0; i < num_rows; i++) if (fabs(h_y_gpu[i] - h_y_cpu[i]) > 1e-3) { jds_ok = 0; break; }

    printf("\nResults (Avg over %d iters):\n", iterations);
    printf("1. CSR Baseline:  %.6f ms (%s)\n", t_csr, csr_ok ? "PASS" : "FAIL");
    printf("2. JDS Optimized: %.6f ms (%s)\n", t_jds, jds_ok ? "PASS" : "FAIL");
    printf("Speedup: %.4fx\n", t_csr / t_jds);

    free(h_x); free(h_y_cpu); free(h_y_gpu);
    free_csr(csr); free_jds(jds);
    cudaFree(d_csr_data); cudaFree(d_csr_indices); cudaFree(d_csr_ptr);
    cudaFree(d_jds_data); cudaFree(d_jds_indices); cudaFree(d_jds_iter_ptr); cudaFree(d_jds_perm);
    cudaFree(d_x); cudaFree(d_y);
    return 0;
}
