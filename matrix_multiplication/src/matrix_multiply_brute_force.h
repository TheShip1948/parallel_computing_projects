#pragma once
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>
#include "matrix_exceptions.h"

// Function declaration
std::vector<std::vector<int>> multiply_matrices_brute_force(
    const std::vector<std::vector<int>>& A, 
    const std::vector<std::vector<int>>& B
);
