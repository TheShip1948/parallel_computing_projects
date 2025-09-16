#include "matrix_multiply_rows.h"
#include <iostream>

// Constructor implementation
MatrixMultiplierRows::MatrixMultiplierRows(const std::vector<std::vector<int>>& a, 
                                   const std::vector<std::vector<int>>& b) 
    : A(a), B(b) {
    rows_A = A.size();
    cols_A = A[0].size();
    cols_B = B[0].size();
    C.resize(rows_A, std::vector<int>(cols_B, 0));
}

// Function to calculate optimal number of threads (as done in C++ Concurrency in Action)
unsigned MatrixMultiplierRows::get_optimal_thread_count(unsigned int min_per_thread) const {
    // Get hardware concurrency (number of cores)
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    
    // Calculate total work (for row-wise division, work is number of rows)
    const unsigned int total_work = rows_A;
    
    // Calculate minimum work per thread
    // const unsigned min_per_thread = 20;

    // Calculate maximum threads based on minimum work per thread
    const unsigned max_threads = (total_work + min_per_thread - 1) / min_per_thread;
    
    // Choose the minimum of hardware threads and max useful threads
    // If hardware_concurrency() returns 0, default to 2
    const unsigned num_threads = std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);
    
    return num_threads;
}

// Method 1: Divide by rows - each thread processes a range of rows
void MatrixMultiplierRows::multiply_by_rows(int start_row, int end_row) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < cols_A; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Parallel multiplication using row-wise division
std::vector<std::vector<int>> MatrixMultiplierRows::parallel_multiply_rows(int num_threads) {
    std::vector<std::thread> threads;
    int rows_per_thread = rows_A / num_threads;
    int remaining_rows = rows_A % num_threads;

    int start_row = 0;
    for (int t = 0; t < num_threads; ++t) {
        int end_row = start_row + rows_per_thread;
        if (t < remaining_rows) {
            end_row++;
        }
        
        threads.emplace_back(&MatrixMultiplierRows::multiply_by_rows, this, start_row, end_row);
        start_row = end_row;
    }

    for (auto& t : threads) {
        t.join();
    }

    return C;
}