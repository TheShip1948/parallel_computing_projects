#include "matrix.h"

// Implementation of multiply_matrices
std::vector<std::vector<int>> multiply_matrices(
    const std::vector<std::vector<int>>& A, 
    const std::vector<std::vector<int>>& B
) {
    if(A.empty() || B.empty()){
        throw MatrixDimensionError("Input matrices cannot be empty.");
    }

    int rowsA = A.size(); 
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size(); 

    if(colsA != rowsB) {
        throw MatrixDimensionError("Number of columns in A must equal number of rows in B.");
    }

    std::vector<std::vector<int>> result(rowsA, std::vector<int>(colsB, 0)); 

    for(int i = 0; i < rowsA; ++i) {
        for(int j = 0; j < colsB; ++j) {
            for(int k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result; 
}

// Implementation of printMatrix
void printMatrix(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}
