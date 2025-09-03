#pragma once
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>

class MatrixDimensionError : public std::exception {
private: 
    std::string message; 
public: 
    MatrixDimensionError(const std::string& msg) : message(msg) {} 
    const char* what() const noexcept override {
        return message.c_str(); 
    }
};

// Function declaration
std::vector<std::vector<int>> multiply_matrices(
    const std::vector<std::vector<int>>& A, 
    const std::vector<std::vector<int>>& B
);

// Helper function
void printMatrix(const std::vector<std::vector<int>>& matrix);