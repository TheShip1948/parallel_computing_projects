
#ifndef MATRIX_MULTIPLIER_ROWS_H
#define MATRIX_MULTIPLIER_ROWS_H

#include <vector>
#include <thread>

class MatrixMultiplierRows {
private:
    std::vector<std::vector<int>> A, B, C;
    int rows_A, cols_A, cols_B;

public:
    // Constructor
    MatrixMultiplierRows(const std::vector<std::vector<int>>& a, 
                     const std::vector<std::vector<int>>& b);

    // Function to calculate optimal number of threads
    unsigned get_optimal_thread_count(unsigned int min_per_thread = 20) const;

    // Method 1: Divide by rows - each thread processes a range of rows
    void multiply_by_rows(int start_row, int end_row);

    // Parallel multiplication using row-wise division
    std::vector<std::vector<int>> parallel_multiply_rows(int num_threads = 4);
};

#endif // MATRIX_MULTIPLIER_ROWS_H