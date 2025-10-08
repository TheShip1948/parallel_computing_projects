#ifndef MATRIX_MULTIPLY_CUDA_H
#define MATRIX_MULTIPLY_CUDA_H

#include <vector>

class MatrixMultiplyCuda {
public:
    // Multiply two matrices using CUDA
    static std::vector<std::vector<int>> multiply(
        const std::vector<std::vector<int>>& A,
        const std::vector<std::vector<int>>& B);
};

#endif // MATRIX_MULTIPLY_CUDA_H
