#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

#include <vector>
#include <stdexcept>
#include "matrix_exceptions.h" 

// Matrix type alias for convenience
using Matrix = std::vector<std::vector<int>>;

class MatrixMultiplierStrassen {
private:
    // Threshold for switching from Strassen to standard multiplication
    static const int STRASSEN_THRESHOLD = 64;
    
    // Helper functions for Strassen algorithm
    static Matrix addMatrices(const Matrix& A, const Matrix& B);
    static Matrix subtractMatrices(const Matrix& A, const Matrix& B);
    static Matrix standardMultiply(const Matrix& A, const Matrix& B);
    static Matrix strassenRecursive(const Matrix& A, const Matrix& B);
    
    // Matrix partitioning and padding functions
    static Matrix padMatrix(const Matrix& matrix, int newSize);
    static Matrix unpadMatrix(const Matrix& matrix, int originalRows, int originalCols);
    static void splitMatrix(const Matrix& matrix, Matrix& topLeft, Matrix& topRight, 
                           Matrix& bottomLeft, Matrix& bottomRight);
    static Matrix combineMatrices(const Matrix& topLeft, const Matrix& topRight,
                                 const Matrix& bottomLeft, const Matrix& bottomRight);
    
    // Matrix validation functions
    static void validateMatrix(const Matrix& matrix, const std::string& name);
    static void validateDimensions(const Matrix& A, const Matrix& B);
    
public:
    // Main multiplication functions
    static Matrix multiplyStandard(const Matrix& A, const Matrix& B);
    static Matrix multiplyStrassen(const Matrix& A, const Matrix& B);
    
    // Utility functions
    static void printMatrix(const Matrix& matrix);
    static Matrix createMatrix(int rows, int cols, int value = 0);
    static int nextPowerOfTwo(int n);
};

#endif // MATRIX_MULTIPLY_H