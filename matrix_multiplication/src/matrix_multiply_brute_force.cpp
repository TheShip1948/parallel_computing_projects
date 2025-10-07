#include "matrix_multiply_brute_force.h"

MatrixMultiplierBruteForce::MatrixMultiplierBruteForce(
    const std::vector<std::vector<int>>& a, 
    const std::vector<std::vector<int>>& b
): A(a), B(b) {
    rows_A = A.size();
    cols_A = A[0].size();
    cols_B = B[0].size();
    C.resize(rows_A, std::vector<int>(cols_B, 0));
}

// Implementation of multiply_matrices
std::vector<std::vector<int>> MatrixMultiplierBruteForce::multiply_matrices_brute_force() {
    if(A.empty() || B.empty()){
        throw MatrixDimensionError("Input matrices cannot be empty.");
    }

    int rows_B = B.size();

    if(cols_A != rows_B) {
        throw MatrixDimensionError("Number of columns in A must equal number of rows in B.");
    }

    for(int i = 0; i < rows_A; ++i) {
        for(int j = 0; j < cols_B; ++j) {
            for(int k = 0; k < cols_A; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C; 
}

