#include <iostream>
#include <algorithm>
#include <cmath>
#include "matrix_multiply_strassen.h"

// MatrixDimensionError implementation
MatrixDimensionError::MatrixDimensionError(const std::string& msg) : message(msg) {}

const char* MatrixDimensionError::what() const noexcept {
    return message.c_str();
}

// Matrix validation functions
void MatrixMultiplierStrassen::validateMatrix(const Matrix& matrix, const std::string& name) {
    if (matrix.empty()) {
        throw MatrixDimensionError("Matrix " + name + " cannot be empty");
    }
    
    int expectedCols = matrix[0].size();
    if (expectedCols == 0) {
        throw MatrixDimensionError("Matrix " + name + " cannot have zero columns");
    }
    
    for (size_t i = 0; i < matrix.size(); i++) {
        if (matrix[i].size() != expectedCols) {
            throw MatrixDimensionError("Matrix " + name + " has inconsistent row sizes at row " + std::to_string(i));
        }
    }
}

void MatrixMultiplierStrassen::validateDimensions(const Matrix& A, const Matrix& B) {
    validateMatrix(A, "A");
    validateMatrix(B, "B");
    
    if (A[0].size() != B.size()) {
        throw MatrixDimensionError(
            "Matrix dimensions incompatible for multiplication. "
            "Matrix A has " + std::to_string(A[0].size()) + " columns, "
            "but Matrix B has " + std::to_string(B.size()) + " rows."
        );
    }
}

// Standard matrix multiplication (O(n^3))
Matrix MatrixMultiplierStrassen::standardMultiply(const Matrix& A, const Matrix& B) {
    validateDimensions(A, B);
    
    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();
    
    Matrix result(rowsA, std::vector<int>(colsB, 0));
    
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return result;
}

// Matrix addition
Matrix MatrixMultiplierStrassen::addMatrices(const Matrix& A, const Matrix& B) {
    int rows = A.size();
    int cols = A[0].size();
    Matrix result(rows, std::vector<int>(cols));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    
    return result;
}

// Matrix subtraction
Matrix MatrixMultiplierStrassen::subtractMatrices(const Matrix& A, const Matrix& B) {
    int rows = A.size();
    int cols = A[0].size();
    Matrix result(rows, std::vector<int>(cols));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    
    return result;
}

// Find next power of 2
int MatrixMultiplierStrassen::nextPowerOfTwo(int n) {
    if (n <= 0) return 1;
    if ((n & (n - 1)) == 0) return n; // Already power of 2
    
    int power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

// Pad matrix to specified size
Matrix MatrixMultiplierStrassen::padMatrix(const Matrix& matrix, int newSize) {
    int originalRows = matrix.size();
    int originalCols = matrix[0].size();
    
    Matrix padded(newSize, std::vector<int>(newSize, 0));
    
    for (int i = 0; i < originalRows; i++) {
        for (int j = 0; j < originalCols; j++) {
            padded[i][j] = matrix[i][j];
        }
    }
    
    return padded;
}

// Remove padding from matrix
Matrix MatrixMultiplierStrassen::unpadMatrix(const Matrix& matrix, int originalRows, int originalCols) {
    Matrix unpadded(originalRows, std::vector<int>(originalCols));
    
    for (int i = 0; i < originalRows; i++) {
        for (int j = 0; j < originalCols; j++) {
            unpadded[i][j] = matrix[i][j];
        }
    }
    
    return unpadded;
}

// Split matrix into 4 quadrants
void MatrixMultiplierStrassen::splitMatrix(const Matrix& matrix, Matrix& topLeft, Matrix& topRight, 
                                  Matrix& bottomLeft, Matrix& bottomRight) {
    int n = matrix.size();
    int half = n / 2;
    
    topLeft = Matrix(half, std::vector<int>(half));
    topRight = Matrix(half, std::vector<int>(half));
    bottomLeft = Matrix(half, std::vector<int>(half));
    bottomRight = Matrix(half, std::vector<int>(half));
    
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            topLeft[i][j] = matrix[i][j];
            topRight[i][j] = matrix[i][j + half];
            bottomLeft[i][j] = matrix[i + half][j];
            bottomRight[i][j] = matrix[i + half][j + half];
        }
    }
}

// Combine 4 quadrants into single matrix
Matrix MatrixMultiplierStrassen::combineMatrices(const Matrix& topLeft, const Matrix& topRight,
                                        const Matrix& bottomLeft, const Matrix& bottomRight) {
    int half = topLeft.size();
    int n = half * 2;
    Matrix combined(n, std::vector<int>(n));
    
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            combined[i][j] = topLeft[i][j];
            combined[i][j + half] = topRight[i][j];
            combined[i + half][j] = bottomLeft[i][j];
            combined[i + half][j + half] = bottomRight[i][j];
        }
    }
    
    return combined;
}

// Recursive Strassen multiplication
Matrix MatrixMultiplierStrassen::strassenRecursive(const Matrix& A, const Matrix& B) {
    int n = A.size();
    
    // Base case: use standard multiplication for small matrices
    if (n <= STRASSEN_THRESHOLD) {
        return standardMultiply(A, B);
    }
    
    // Split matrices into quadrants
    Matrix A11, A12, A21, A22;
    Matrix B11, B12, B21, B22;
    
    splitMatrix(A, A11, A12, A21, A22);
    splitMatrix(B, B11, B12, B21, B22);
    
    // Calculate the 7 Strassen products
    Matrix M1 = strassenRecursive(addMatrices(A11, A22), addMatrices(B11, B22));
    Matrix M2 = strassenRecursive(addMatrices(A21, A22), B11);
    Matrix M3 = strassenRecursive(A11, subtractMatrices(B12, B22));
    Matrix M4 = strassenRecursive(A22, subtractMatrices(B21, B11));
    Matrix M5 = strassenRecursive(addMatrices(A11, A12), B22);
    Matrix M6 = strassenRecursive(subtractMatrices(A21, A11), addMatrices(B11, B12));
    Matrix M7 = strassenRecursive(subtractMatrices(A12, A22), addMatrices(B21, B22));
    
    // Calculate result quadrants
    Matrix C11 = addMatrices(subtractMatrices(addMatrices(M1, M4), M5), M7);
    Matrix C12 = addMatrices(M3, M5);
    Matrix C21 = addMatrices(M2, M4);
    Matrix C22 = addMatrices(subtractMatrices(addMatrices(M1, M3), M2), M6);
    
    // Combine quadrants
    return combineMatrices(C11, C12, C21, C22);
}

// Public interface for Strassen multiplication
Matrix MatrixMultiplierStrassen::multiplyStrassen(const Matrix& A, const Matrix& B) {
    validateDimensions(A, B);
    
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();
    
    // For Strassen algorithm, we need square matrices with power-of-2 dimensions
    int maxDim = std::max({rowsA, colsA, rowsB, colsB});
    int paddedSize = nextPowerOfTwo(maxDim);
    
    // Pad matrices to required size
    Matrix paddedA = padMatrix(A, paddedSize);
    Matrix paddedB = padMatrix(B, paddedSize);
    
    // Perform Strassen multiplication
    Matrix paddedResult = strassenRecursive(paddedA, paddedB);
    
    // Remove padding and return result
    return unpadMatrix(paddedResult, rowsA, colsB);
}
