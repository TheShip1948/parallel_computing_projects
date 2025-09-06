#include "matrix_multiply_blocks.h"
#include <iostream>
#include <algorithm>
#include <cmath>

// Constructor implementation
MatrixMultiplierBlocks::MatrixMultiplierBlocks(const std::vector<std::vector<int>>& a, 
                                               const std::vector<std::vector<int>>& b) 
    : A(a), B(b) {
    rows_A = A.size();
    cols_A = A[0].size();
    cols_B = B[0].size();
    C.resize(rows_A, std::vector<int>(cols_B, 0));
}

// Function to calculate optimal number of threads (as done in C++ Concurrency in Action)
unsigned int MatrixMultiplierBlocks::get_optimal_thread_count(unsigned int min_per_thread) const {
    // Get hardware concurrency (number of cores)
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    
    // Calculate total work (for block-wise division, work is total result matrix elements)
    const unsigned int total_work = rows_A * cols_B;
    
    // Calculate maximum threads based on minimum work per thread
    const unsigned int max_threads = (total_work + min_per_thread - 1) / min_per_thread;
    
    // Choose the minimum of hardware threads and max useful threads
    // If hardware_concurrency() returns 0, default to 2
    const unsigned int num_threads = std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);
    
    return num_threads;
}

// Method 3: Divide by blocks - each thread processes a 2D block of the result matrix
void MatrixMultiplierBlocks::multiply_block(int start_row, int end_row, int start_col, int end_col) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = start_col; j < end_col; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < cols_A; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Parallel multiplication using block-wise division with specified thread count
std::vector<std::vector<int>> MatrixMultiplierBlocks::parallel_multiply_blocks(int num_threads) {
    std::vector<std::thread> threads;
    
    // Calculate block dimensions
    int blocks_per_dim = static_cast<int>(std::sqrt(num_threads));
    if (blocks_per_dim * blocks_per_dim < num_threads) {
        blocks_per_dim++;
    }
    
    int row_block_size = (rows_A + blocks_per_dim - 1) / blocks_per_dim;
    int col_block_size = (cols_B + blocks_per_dim - 1) / blocks_per_dim;
    
    int thread_count = 0;
    for (int block_row = 0; block_row < blocks_per_dim && thread_count < num_threads; ++block_row) {
        for (int block_col = 0; block_col < blocks_per_dim && thread_count < num_threads; ++block_col) {
            int start_row = block_row * row_block_size;
            int end_row = std::min(start_row + row_block_size, rows_A);
            int start_col = block_col * col_block_size;
            int end_col = std::min(start_col + col_block_size, cols_B);
            
            if (start_row < rows_A && start_col < cols_B) {
                threads.emplace_back(&MatrixMultiplierBlocks::multiply_block, this, 
                                   start_row, end_row, start_col, end_col);
                thread_count++;
            }
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    return C;
}

// Alternative block division using futures with specified thread count
std::vector<std::vector<int>> MatrixMultiplierBlocks::parallel_multiply_blocks_futures(int num_threads) {
    std::vector<std::future<void>> futures;
    
    // Calculate optimal block dimensions
    int blocks_per_dim = static_cast<int>(std::sqrt(num_threads));
    if (blocks_per_dim * blocks_per_dim < num_threads) {
        blocks_per_dim++;
    }
    
    int row_block_size = (rows_A + blocks_per_dim - 1) / blocks_per_dim;
    int col_block_size = (cols_B + blocks_per_dim - 1) / blocks_per_dim;
    
    for (int block_row = 0; block_row < blocks_per_dim; ++block_row) {
        for (int block_col = 0; block_col < blocks_per_dim; ++block_col) {
            int start_row = block_row * row_block_size;
            int end_row = std::min(start_row + row_block_size, rows_A);
            int start_col = block_col * col_block_size;
            int end_col = std::min(start_col + col_block_size, cols_B);
            
            if (start_row < rows_A && start_col < cols_B) {
                futures.push_back(
                    std::async(std::launch::async, 
                             &MatrixMultiplierBlocks::multiply_block, this,
                             start_row, end_row, start_col, end_col)
                );
            }
        }
    }

    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }

    return C;
}
