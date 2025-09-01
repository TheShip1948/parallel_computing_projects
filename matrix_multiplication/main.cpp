#include <iostream> 
#include <vector> 
#include <stdexcept> 

// Custom exception class for matrix dimension error 
class MatrixDimensionError : public std::exception {
private: 
    std::string message; 

public: 
    MatrixDimensionError(const std::string& msg) : message(msg) {} 
    const char* what() const noexcept override {
        return message.c_str(); 
    }
};


// Matrix multiplication function - Brute Force Implementation 
std::vector<std::vector<int>> multiply_matrices(
    const std::vector<std::vector<int>>& A, 
    const std::vector<std::vector<int>>& B
) {
    // Check if matrices are empty 
    if(A.empty() || B.empty()){
        throw MatrixDimensionError("Input matrices cannot be empty.");
    }

    // Check if multiplication is possible (columns of A must equal rows of B) 
    int rowsA = A.size(); 
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size(); 

    if(colsA != rowsB) {
        throw MatrixDimensionError("Number of columns in A must equal number of rows in B.");
    }

    // Initialize result matrix with zeros 
    std::vector<std::vector<int>> result(rowsA, std::vector<int>(colsB, 0)); 

    // Brute force matrix multiplication O(n^3) 
    for(int i = 0; i < rowsA; ++i) {
        for(int j = 0; j < colsB; ++j) {
            for(int k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result; 
}

// Helper function to print a matrix
void printMatrix(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}


// Example usage and testing
int main() {
    try {
        // Example 1: Valid multiplication
        std::vector<std::vector<int>> A = {
            {1, 2, 3},
            {4, 5, 6}
        };
        
        std::vector<std::vector<int>> B = {
            {7, 8},
            {9, 10},
            {11, 12}
        };
        
        std::cout << "Matrix A (2x3):" << std::endl;
        printMatrix(A);
        
        std::cout << "\nMatrix B (3x2):" << std::endl;
        printMatrix(B);
        
        auto result = multiply_matrices(A, B);
        
        std::cout << "\nResult A * B (2x2):" << std::endl;
        printMatrix(result);
        
        // Example 2: Invalid dimensions - this will throw an exception
        std::cout << "\n--- Testing invalid dimensions ---" << std::endl;
        
        std::vector<std::vector<int>> C = {
            {1, 2},
            {3, 4}
        };
        
        std::vector<std::vector<int>> D = {
            {5, 6, 7},
            {8, 9, 10},
            {11, 12, 13}
        };
        
        std::cout << "Attempting to multiply 2x2 matrix with 3x3 matrix..." << std::endl;
        auto invalidResult = multiply_matrices(C, D);
        
    } catch (const MatrixDimensionError& e) {
        std::cerr << "Matrix Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "General Error: " << e.what() << std::endl;
    }
    
    return 0;
}