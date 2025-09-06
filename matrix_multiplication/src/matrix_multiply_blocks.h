#ifndef MATRIX_MULTIPLIER_BLOCKS_H
#define MATRIX_MULTIPLIER_BLOCKS_H

#include <vector>
#include <thread>
#include <future>

class MatrixMultiplierBlocks {
private:
    std::vector<std::vector<int>> A, B, C;
    int rows_A, cols_A, cols_B;

    // Default minimum number of elements per thread
    static const unsigned int default_min_per_thread = 25;

public:
    // Constructor
    MatrixMultiplierBlocks(const std::vector<std::vector<int>>& a, 
                           const std::vector<std::vector<int>>& b);

    // Function to calculate optimal number of threads
    unsigned int get_optimal_thread_count(unsigned int min_per_thread = default_min_per_thread) const;

    // Method 3: Divide by blocks - each thread processes a 2D block of the result matrix
    void multiply_block(int start_row, int end_row, int start_col, int end_col);

    // Parallel multiplication using block-wise division with specified thread count
    std::vector<std::vector<int>> parallel_multiply_blocks(int num_threads);

    // Alternative block division using futures with specified thread count
    std::vector<std::vector<int>> parallel_multiply_blocks_futures(int num_threads);

};

#endif // MATRIX_MULTIPLIER_BLOCKS_H