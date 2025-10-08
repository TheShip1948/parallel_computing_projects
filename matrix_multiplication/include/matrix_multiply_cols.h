#ifndef MATRIX_MULTIPLIER_COLS_H
#define MATRIX_MULTIPLIER_COLS_H

#include <vector>
#include <thread>
#include <mutex>

class MatrixMultiplierCols {
private:
    std::vector<std::vector<int>> A, B, C;
    int rows_A, cols_A, cols_B;
    std::mutex mtx; // For thread-safe output access if needed

    // Default minimum number of elements per thread
    static const unsigned int default_min_per_thread = 25;

public:
    // Constructor
    MatrixMultiplierCols(const std::vector<std::vector<int>>& a, 
                         const std::vector<std::vector<int>>& b);

    // Function to calculate optimal number of threads
    unsigned int get_optimal_thread_count(unsigned int min_per_thread = default_min_per_thread) const;

    // Method 2: Divide by columns - each thread processes a range of columns
    void multiply_by_cols(int start_col, int end_col);

    // Parallel multiplication using column-wise division with specified thread count 
    std::vector<std::vector<int>> parallel_multiply_cols(int num_threads);
};

#endif // MATRIX_MULTIPLIER_COLS_H