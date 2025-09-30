#include "matrix_multiplier_cacheoblivious.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>

// Constructor implementation: copies inputs and validates shapes.
MatrixMultiplierCacheOblivious::MatrixMultiplierCacheOblivious(const std::vector<std::vector<int>>& a,
                                                               const std::vector<std::vector<int>>& b)
    : A(a), B(b)
{
    if (a.empty() || b.empty()) {
        throw std::invalid_argument("Input matrices must not be empty.");
    }
    if (a[0].empty() || b[0].empty()) {
        throw std::invalid_argument("Input matrices must have at least one column.");
    }
    // All rows in A must have same length, all rows in B same length.
    size_t a_cols = a[0].size();
    for (const auto& row : a) if (row.size() != a_cols) throw std::invalid_argument("Irregular row size in A.");
    size_t b_cols = b[0].size();
    for (const auto& row : b) if (row.size() != b_cols) throw std::invalid_argument("Irregular row size in B.");

    if (a_cols != b.size()) {
        throw std::invalid_argument("Number of columns in A must equal number of rows in B.");
    }

    N = static_cast<int>(A.size());
    K = static_cast<int>(a_cols);
    M = static_cast<int>(b_cols);

    // Initialize result matrix C with correct final dimensions (un-padded).
    C.assign(N, std::vector<int>(M, 0));
}

// Choose an effective thread count given a minimum work per thread. Caps at hardware and 8.
unsigned int MatrixMultiplierCacheOblivious::get_optimal_thread_count(unsigned int min_per_thread) const {
    const unsigned int hardware_threads = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 2u;
    const unsigned int total_work = static_cast<unsigned int>(N) * static_cast<unsigned int>(M);
    const unsigned int max_threads = (total_work + min_per_thread - 1) / min_per_thread;
    const unsigned int num_threads = std::min(hardware_threads, max_threads);
    return std::min(num_threads, 8u); // cap to 8 top-level tasks
}

// Multiply a base block for padded square matrices Ap, Bp, writing into Cp.
// orig_N/K/M are original (unpadded) dimensions used to avoid writing outside target region
void MatrixMultiplierCacheOblivious::multiply_base_padded(
    const std::vector<std::vector<int>>& Ap,
    const std::vector<std::vector<int>>& Bp,
    std::vector<std::vector<int>>& Cp,
    int a_sr, int a_sc,
    int b_sr, int b_sc,
    int c_sr, int c_sc,
    int size,
    int orig_N, int orig_K, int orig_M,
    bool add_to_C)
{
    // Determine the actual bounds (so we don't write outside original result area).
    // For padded arrays Ap and Bp we assume they are at least size x size at the provided start.
    int i_max = size;
    int j_max = size;
    int k_max = size;

    // But clip loops to original matrix dimensions relative to c_sr/c_sc for writing:
    int max_write_rows = std::max(0, orig_N - c_sr); // how many real rows remain for this c block
    int max_write_cols = std::max(0, orig_M - c_sc);

    int block_rows = std::min(i_max, max_write_rows);
    int block_cols = std::min(j_max, max_write_cols);
    int block_k = std::min(k_max, std::max(0, orig_K - a_sc));

    for (int i = 0; i < block_rows; ++i) {
        for (int j = 0; j < block_cols; ++j) {
            int sum = 0;
            for (int k = 0; k < block_k; ++k) {
                sum += Ap[a_sr + i][a_sc + k] * Bp[b_sr + k][b_sc + j];
            }
            if (add_to_C) {
                Cp[c_sr + i][c_sc + j] += sum;
            } else {
                Cp[c_sr + i][c_sc + j] = sum;
            }
        }
    }
}

// Recursive padded multiply. Works on size x size square blocks, where size is power-of-two.
void MatrixMultiplierCacheOblivious::multiply_recursive_padded(
    const std::vector<std::vector<int>>& Ap,
    const std::vector<std::vector<int>>& Bp,
    std::vector<std::vector<int>>& Cp,
    int a_sr, int a_sc,
    int b_sr, int b_sc,
    int c_sr, int c_sc,
    int size,
    int orig_N, int orig_K, int orig_M,
    bool add_to_C)
{
    if (size <= BASE_SIZE) {
        multiply_base_padded(Ap, Bp, Cp, a_sr, a_sc, b_sr, b_sc, c_sr, c_sc, size, orig_N, orig_K, orig_M, add_to_C);
        return;
    }

    int half = size / 2;

    // First 4 multiplications (C = A*B) — they may overwrite or set
    multiply_recursive_padded(Ap, Bp, Cp, a_sr, a_sc, b_sr, b_sc, c_sr, c_sc, half, orig_N, orig_K, orig_M, add_to_C);
    multiply_recursive_padded(Ap, Bp, Cp, a_sr, a_sc, b_sr, b_sc + half, c_sr, c_sc + half, half, orig_N, orig_K, orig_M, add_to_C);
    multiply_recursive_padded(Ap, Bp, Cp, a_sr + half, a_sc, b_sr, b_sc, c_sr + half, c_sc, half, orig_N, orig_K, orig_M, add_to_C);
    multiply_recursive_padded(Ap, Bp, Cp, a_sr + half, a_sc, b_sr, b_sc + half, c_sr + half, c_sc + half, half, orig_N, orig_K, orig_M, add_to_C);

    // Second 4 multiplications (C += A*B)
    multiply_recursive_padded(Ap, Bp, Cp, a_sr, a_sc + half, b_sr + half, b_sc, c_sr, c_sc, half, orig_N, orig_K, orig_M, true);
    multiply_recursive_padded(Ap, Bp, Cp, a_sr, a_sc + half, b_sr + half, b_sc + half, c_sr, c_sc + half, half, orig_N, orig_K, orig_M, true);
    multiply_recursive_padded(Ap, Bp, Cp, a_sr + half, a_sc + half, b_sr + half, b_sc, c_sr + half, c_sc, half, orig_N, orig_K, orig_M, true);
    multiply_recursive_padded(Ap, Bp, Cp, a_sr + half, a_sc + half, b_sr + half, b_sc + half, c_sr + half, c_sc + half, half, orig_N, orig_K, orig_M, true);
}

// Public: parallel cache-oblivious multiply.
// We pad to a square matrix of size = next power-of-two >= max(N, K, M), then run the recursive algorithm.
// num_threads is only a hint (we cap to 8 top-level tasks).
std::vector<std::vector<int>> MatrixMultiplierCacheOblivious::parallel_multiply_cacheoblivious(int num_threads) {
    // Determine padded size (power of two >= max dimension)
    int max_dim = std::max({N, K, M});
    int size = 1;
    while (size < max_dim) size <<= 1;

    // Prepare padded matrices Ap, Bp, Cp (size x size)
    std::vector<std::vector<int>> Ap(size, std::vector<int>(size, 0));
    std::vector<std::vector<int>> Bp(size, std::vector<int>(size, 0));
    std::vector<std::vector<int>> Cp(size, std::vector<int>(size, 0));

    // Copy A into Ap and B into Bp
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            Ap[i][j] = A[i][j];
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < M; ++j) {
            Bp[i][j] = B[i][j];
        }
    }

    // Cap the top-level parallelism (we have 8 top-level tasks)
    int effective_threads = std::min(num_threads, 8);

    // Partition into 8 tasks (like your earlier design)
    int half = size / 2;

    std::vector<std::tuple<int,int,int,int,int,int,bool>> tasks = {
        // First 4 (set)
        {0, 0, 0, 0, 0, 0, false},
        {0, 0, 0, half, 0, half, false},
        {half, 0, 0, 0, half, 0, false},
        {half, 0, 0, half, half, half, false},
        // Last 4 (add)
        {0, half, half, 0, 0, 0, true},
        {0, half, half, half, 0, half, true},
        {half, half, half, 0, half, 0, true},
        {half, half, half, half, half, half, true}
    };

    // If user requested single-thread, execute sequentially.
    if (effective_threads <= 1) {
        for (const auto& t : tasks) {
            multiply_recursive_padded(Ap, Bp, Cp,
                                      std::get<0>(t), std::get<1>(t),
                                      std::get<2>(t), std::get<3>(t),
                                      std::get<4>(t), std::get<5>(t),
                                      half,
                                      N, K, M,
                                      std::get<6>(t));
        }
    } else {
        // Launch 8 tasks with std::async (runtime will manage concurrency).
        std::vector<std::future<void>> futures;
        futures.reserve(tasks.size());
        for (const auto& t : tasks) {
            futures.emplace_back(std::async(std::launch::async,
                [&Ap, &Bp, &Cp, t, half, this]() {
                    multiply_recursive_padded(Ap, Bp, Cp,
                                              std::get<0>(t), std::get<1>(t),
                                              std::get<2>(t), std::get<3>(t),
                                              std::get<4>(t), std::get<5>(t),
                                              half,
                                              this->N, this->K, this->M,
                                              std::get<6>(t));
                }
            ));
        }
        for (auto &f : futures) f.get();
    }

    // Copy the relevant top-left N x M region of Cp into result C and return it.
    std::vector<std::vector<int>> result(N, std::vector<int>(M, 0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            result[i][j] = Cp[i][j];
        }
    }

    // update internal C (optional)
    C = result;
    return result;
}
