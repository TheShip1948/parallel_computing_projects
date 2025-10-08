#ifndef MAT_MUL_CACHE_OBLIVIOUS_CLASS_HEADER_2025_V1
#define MAT_MUL_CACHE_OBLIVIOUS_CLASS_HEADER_2025_V1

#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <future>
#include <tuple>
#include <stdexcept>

class MatrixMultiplierCacheOblivious {
private:
    // Original user-provided matrices (un-padded)
    std::vector<std::vector<int>> A, B, C;
    int N, K, M; // Rows_A (N), Cols_A/Rows_B (K), Cols_B (M)

    // Base case size for the recursion.
    static const int BASE_SIZE = 16;

    // Default minimum number of elements per thread
    static const unsigned int default_min_per_thread = 256;

    // Internal (padded) matrices used by the recursive algorithm.
    // These are created inside parallel_multiply_cacheoblivious and passed to the static helpers.
    // (Not stored persistently between calls.)
    // The core block multiply for the padded matrices (operates on given start indices).
    static void multiply_base_padded(const std::vector<std::vector<int>>& Ap,
                                     const std::vector<std::vector<int>>& Bp,
                                     std::vector<std::vector<int>>& Cp,
                                     int a_sr, int a_sc,
                                     int b_sr, int b_sc,
                                     int c_sr, int c_sc,
                                     int size,
                                     int orig_N, int orig_K, int orig_M,
                                     bool add_to_C);

    // Recursive padded multiply (square blocks of size power-of-two).
    static void multiply_recursive_padded(const std::vector<std::vector<int>>& Ap,
                                          const std::vector<std::vector<int>>& Bp,
                                          std::vector<std::vector<int>>& Cp,
                                          int a_sr, int a_sc,
                                          int b_sr, int b_sc,
                                          int c_sr, int c_sc,
                                          int size,
                                          int orig_N, int orig_K, int orig_M,
                                          bool add_to_C);

public:
    // Constructor
    MatrixMultiplierCacheOblivious(const std::vector<std::vector<int>>& a,
                                   const std::vector<std::vector<int>>& b);

    // Function to calculate optimal number of threads
    unsigned int get_optimal_thread_count(unsigned int min_per_thread = default_min_per_thread) const;

    // The main public method for parallel cache-oblivious multiplication.
    // num_threads is a hint; internally we cap the top-level parallelism to at most 8 tasks.
    std::vector<std::vector<int>> parallel_multiply_cacheoblivious(int num_threads);
};

#endif // MAT_MUL_CACHE_OBLIVIOUS_CLASS_HEADER_2025_V1
