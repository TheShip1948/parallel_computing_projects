# ECE408: Applied Parallel Computing Course Labs

This repository contains implementations for the Machine Problems (MPs) of the ECE408 course (Applied Parallel Computing). These labs focus on developing and optimizing parallel algorithms using the CUDA programming model.

## Lab Directory

1.  **[MP1: Vector Addition](./MP1_vector_addition/README.md)** - Basic CUDA setup and vector operations.
2.  **[MP2: Dense Matrix Multiplication](./MP2_dense_matrix_multiplication/README.md)** - Fundamental matrix algorithms and 2D thread indexing.
3.  **[MP3: Tiled Dense Matrix Multiplication](./MP3_tiled_dense_matrix_multiplication/README.md)** - Performance optimization using shared memory tiling.
4.  **[MP4: Tiled 2D Convolution](./MP4_tiled_2D_convolution/README.md)** - Constant memory and shared memory tiling for image processing.
5.  **[MP5: 1D List Reduction](./MP5_1D_list_reduction/README.md)** - Parallel reduction algorithm and warp efficiency.
6.  **[MP6: 1D List Scan](./MP6_1D_list_scan/README.md)** - Work-efficient parallel prefix sum (Scan).
7.  **[MP7: Image Histogram Equalization](./MP7_image_histogram_equalization_algo/README.md)** - Full image processing pipeline with atomic operations and CDF.
8.  **[MP8: Sparse Matrix-Vector Multiplication (SpMV)](./MP8_SpMV_JDS/README.md)** - Optimizing irregular data structures (CSR vs. JDS).

## References
- Course Material: [QiqianFu/ECE408](https://github.com/QiqianFu/ECE408)
- Textbook: *Programming Massively Parallel Processors* by David B. Kirk and Wen-mei W. Hwu.
