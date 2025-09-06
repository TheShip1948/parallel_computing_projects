#include "matrix_multiply_cols.h" 
#include <iostream>
#include <algorithm>

// Constructor implementation
MatrixMultiplierCols::MatrixMultiplierCols(const std::vector<std::vector<int>>& a, 
                                           const std::vector<std::vector<int>>& b) 
    : A(a), B(b) {
    rows_A = A.size();
    cols_A = A[0].size();
    cols_B = B[0].size();
    C.resize(rows_A, std::vector<int>(cols_B, 0));
}

// Function to calculate optimal number of threads (as done in C++ Concurrency in Action)
unsigned int MatrixMultiplierCols::get_optimal_thread_count(unsigned int min_per_thread) const {
    // Get hardware concurrency (number of cores)
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    
    // Calculate total work (for column-wise division, work is number of columns)
    const unsigned int total_work = cols_B;
    
    // Calculate maximum threads based on minimum work per thread
    const unsigned int max_threads = (total_work + min_per_thread - 1) / min_per_thread;
    
    // Choose the minimum of hardware threads and max useful threads
    // If hardware_concurrency() returns 0, default to 2
    const unsigned int num_threads = std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);
    
    return num_threads;
}

// Method 2: Divide by columns - each thread processes a range of columns
void MatrixMultiplierCols::multiply_by_cols(int start_col, int end_col) {
    for (int j = start_col; j < end_col; ++j) {
        for (int i = 0; i < rows_A; ++i) {
            C[i][j] = 0;
            for (int k = 0; k < cols_A; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Parallel multiplication using column-wise division with specified thread count
std::vector<std::vector<int>> MatrixMultiplierCols::parallel_multiply_cols(int num_threads) {
    std::vector<std::thread> threads;
    int cols_per_thread = cols_B / num_threads;
    int remaining_cols = cols_B % num_threads;

    int start_col = 0;
    for (int t = 0; t < num_threads; ++t) {
        int end_col = start_col + cols_per_thread;
        if (t < remaining_cols) {
            end_col++;
        }
        
        threads.emplace_back(&MatrixMultiplierCols::multiply_by_cols, this, start_col, end_col);
        start_col = end_col;
    }

    for (auto& t : threads) {
        t.join();
    }

    return C;
}
