#pragma once
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>
#include "matrix_exceptions.h"

class MatrixMultiplierBruteForce {
private:
    std::vector<std::vector<int>> A, B, C; 
    int rows_A, cols_A, cols_B; 

public: 
    MatrixMultiplierBruteForce(
        const std::vector<std::vector<int>>& a, 
        const std::vector<std::vector<int>>& b
        );
    std::vector<std::vector<int>> multiply_matrices_brute_force();
};


